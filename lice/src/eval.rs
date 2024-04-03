use crate::{
    combinator::{Combinator, Reduce, ReduxCode},
    ffi::{self, FfiSymbol, ForeignPtr},
    float::{FShow, Float},
    integer::Integer,
    memory::{App, FromValue, IntoValue, Pointer, Value},
    string::{eval_hstring, new_castringlen, peek_castring, peek_castringlen},
    tick::{Tick, TickError, TickInfo, TickTable},
};
use bitvec::prelude::*;
use core::{
    cell::OnceCell,
    fmt::{self, Debug, Display},
};
use either::{for_both, Either};
use gc_arena::{Arena, Collect, Mutation, Rootable};
use log::debug;

/// A runtime error, probably due to a malformed graph.
///
/// Defined as a separate macro here so that deliberate runtime crashes can be easily searched and refactored.
macro_rules! runtime_crash {
    ($($tokens:tt)*) => {
        runtime_crash(format_args!($($tokens)*))
    };
}

/// An expected failure.
///
/// Written as a separate function so as to attach the `#[cold]` attribute.
#[cold]
fn runtime_crash(args: fmt::Arguments) -> ! {
    panic!("Runtime crash: {args}")
}

type Operator = Either<Combinator, FfiSymbol>;
impl From<Combinator> for Operator {
    fn from(value: Combinator) -> Self {
        Either::Left(value)
    }
}
impl From<FfiSymbol> for Operator {
    fn from(value: FfiSymbol) -> Self {
        Either::Right(value)
    }
}

#[derive(thiserror::Error, Debug)]
pub enum SpineError {
    #[error("could not pop, spine empty")]
    NoPop,
    #[error("could not peek, index {0} is out of bounds")]
    NoPeek(usize),
}

#[derive(Collect, Default)]
#[collect(no_drop)]
struct Spine<'gc> {
    stack: Vec<Pointer<'gc>>,
    /// Invariant: at all times, `watermark <= stack.len()`.
    watermark: usize,
}

impl<'gc> Spine<'gc> {
    fn push(&mut self, app: Pointer<'gc>) {
        self.stack.push(app);
    }

    fn pop(&mut self) -> Result<Pointer<'gc>, SpineError> {
        self.stack.pop().ok_or(SpineError::NoPop)
    }

    fn peek(&self, idx: usize) -> Result<Pointer<'gc>, SpineError> {
        let idx = self.stack.len() - 1 - idx;
        self.stack.get(idx).cloned().ok_or(SpineError::NoPeek(idx))
    }

    #[allow(dead_code)]
    fn has_args(&self, num: usize) -> bool {
        self.watermark <= self.stack.len() - num
    }

    fn save(&mut self) -> usize {
        let prev = self.watermark;
        self.watermark = self.stack.len();
        prev
    }

    fn restore(&mut self, prev: usize) {
        debug_assert!(self.watermark <= self.stack.len());
        self.stack.drain(self.watermark..);
        self.watermark = prev;
    }
}

/// Pending strict evaluation some combinator's arguments.
#[derive(Debug, Collect, Clone, Copy)]
#[collect(require_static)]
struct StrictWork {
    strict: Operator,
    args: BitArr!(for Self::MAX_ARGS),
    watermark: usize,
}

impl StrictWork {
    const MAX_ARGS: usize = 4;

    fn init<'gc>(op: Operator, spine: &mut Spine<'gc>) -> Option<(Self, Pointer<'gc>)> {
        // Figure out which args need to be evaluated, i.e., are not yet WHNF
        // Default state: we don't need to evalute each arg
        let mut args = BitArray::new([0]);

        // Also find index of first argument that needs evaluating
        let first = OnceCell::new();

        for (idx, is_strict) in for_both!(op, o => o.strictness()).iter().enumerate() {
            let app = spine
                .peek(idx)
                .expect("all args should be in spine when starting strict work")
                .follow();
            if *is_strict && !app.unpack_arg().follow().unpack().whnf() {
                // Remember the first argument that needs evaluating:
                let is_pending = first.set(idx).is_err();
                if is_pending {
                    // If first.set().is_err(), then this argument is pending after `arg[first]`
                    // is evaluated, so we mark it as such:
                    args.set(idx, true);
                }
            }
        }
        let first = *first.get()?; // If no args need to be evaluated, then return None
        Some((
            Self {
                strict: op,
                args,
                watermark: spine.save(),
            },
            spine.peek(first).unwrap().follow().unpack_arg().follow(),
        ))
    }

    /// Advance the pending arg state.
    fn advance<'gc>(&mut self, spine: &mut Spine<'gc>) -> Option<Pointer<'gc>> {
        // No need to iterate over index 0 because it should never have been set.
        for idx in 1..Self::MAX_ARGS {
            if self.args[idx] {
                self.args.set(idx, false);

                let app = spine
                    .peek(idx)
                    .expect("all args should be in spine when resuming strict work")
                    .follow();
                let arg = app.unpack_arg().follow();
                if !arg.unpack().whnf() {
                    // Only evaluate this arg next if it is not in WHNF yet
                    return Some(app.unpack_arg().follow());
                }
            }
        }
        spine.restore(self.watermark);
        None
    }
}

#[derive(Clone, Copy, Collect, Debug)]
#[collect(require_static)]
pub enum IO {
    Bind { cc: bool },
    Then,
    Perform,
}

impl Display for IO {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IO::Bind { cc } => {
                if *cc {
                    f.write_str("CC' (>>=)")
                } else {
                    f.write_str("(>>=)")
                }
            }
            IO::Then => f.write_str("(>>)"),
            IO::Perform => f.write_str("PerformIO"),
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum VMError {
    #[error("nothing to reduce")]
    AlreadyDone,
    #[error(transparent)]
    TickError(#[from] TickError),
    #[error("could not resume after tick {tick:?}")]
    TickResume { tick: TickInfo, source: SpineError },
    #[error("missing FFI symbol: {0}")]
    BadDyn(String),
    #[error("could not resume {io}")]
    ResumeIO { io: IO, source: SpineError },
}

type VMResult<T> = Result<T, VMError>;

#[derive(Collect)]
#[collect(no_drop)]
struct State<'gc> {
    tip: Pointer<'gc>,
    spine: Spine<'gc>,
    strict_context: Vec<StrictWork>,
    io_context: Vec<IO>,
    tick_table: TickTable,
}

impl<'gc> State<'gc> {
    fn new(top: Pointer<'gc>, tick_table: TickTable) -> Self {
        Self {
            tip: top,
            spine: Default::default(),
            strict_context: Default::default(),
            io_context: Default::default(),
            tick_table,
        }
    }

    /// Symbolic handler for combinators with well-defined "redux code".
    ///
    /// With luck, the `rustc` optimizer will unroll this stack machine interpreter into efficient
    /// code, such that the `args` are kept in registers, and the `stk` is completely gone.
    #[inline(always)]
    #[track_caller]
    fn do_redux(&mut self, mc: &Mutation<'gc>, comb: Combinator) -> Result<Pointer<'gc>, VMError> {
        #[cfg(debug_assertions)]
        macro_rules! unwrap {
            ($e:expr) => {
                ($e).unwrap()
            };
        }
        #[cfg(not(debug_assertions))]
        macro_rules! unwrap {
            ($e:expr) => {
                unsafe { ($e).unwrap_unchecked() }
            };
        }

        let code = unwrap!(comb.redux());

        let mut top = self.tip;
        let mut args = heapless::Vec::<_, 8>::new();
        let mut stk = heapless::Vec::<_, 8>::new();

        for _ in 0..comb.arity() {
            // TODO: better debug
            let app = self.spine.pop().expect("FIXME");
            unwrap!(args.push(app.unpack_arg()));
            top = app;
        }

        for b in &code[..code.len() - 1] {
            match b {
                ReduxCode::Arg(i) => unwrap!(stk.push(args[*i])),
                ReduxCode::Top => unwrap!(stk.push(top)),
                ReduxCode::App => {
                    let arg = unwrap!(stk.pop());
                    let fun = unwrap!(stk.pop());
                    unwrap!(stk.push(Pointer::new(mc, App { fun, arg })))
                }
            }
        }

        // We need to handle the last argument differently
        match code[code.len() - 1] {
            ReduxCode::Arg(i) => {
                top.set(mc, Value::Ref(args[i]));
                Ok(args[i])
            }
            ReduxCode::Top => {
                unreachable!("no rule should reduce to '^' alone ")
            }
            ReduxCode::App => {
                let arg = unwrap!(stk.pop());
                let fun = unwrap!(stk.pop());
                top.set(mc, App { fun, arg });
                Ok(top)
            }
        }
    }

    fn start_strict(&mut self, op: Operator, mc: &Mutation<'gc>) -> Result<Pointer<'gc>, VMError> {
        if !self.spine.has_args(for_both!(op, o => o.arity())) {
            return self.resume_strict(mc);
        }

        if let Some((work, first)) = StrictWork::init(op, &mut self.spine) {
            // `next` has been reduced to a combinator, though one or more of its arguments
            // may still need to be evaluated before it can be handled
            self.strict_context.push(work);
            Ok(first)
        } else {
            // If start_strict returns None, then we can directly handle the combinator
            match op {
                Either::Left(comb) => self.eval_comb(comb, mc),
                Either::Right(ffi) => self.eval_ffi(ffi, mc),
            }
        }
    }

    fn resume_strict(&mut self, mc: &Mutation<'gc>) -> VMResult<Pointer<'gc>> {
        // `tip` has been reduced to a value, probably because some strict operator had
        // previously demanded strict arguments. We check our strict continuation context
        // to see if we still need to evaluate more arguments, or if we can now handle the
        // strict operator.
        let Some(work) = self.strict_context.last_mut() else {
            runtime_crash!("No pending strict operation");
        };

        if let Some(next) = work.advance(&mut self.spine) {
            debug!(
                "Resuming evaluation of strict {:?}, advanced to next argument",
                work.strict
            );
            Ok(next)
        } else {
            debug!(
                "Resuming evaluation of strict {:?}, all arguments in WHNF",
                work.strict
            );
            let strict = work.strict;
            self.strict_context.pop();
            match strict {
                Either::Left(comb) => self.eval_comb(comb, mc),
                Either::Right(ffi) => self.eval_ffi(ffi, mc),
            }
        }
    }

    fn start_io(&mut self, mc: &Mutation<'gc>, io: IO) -> Result<Pointer<'gc>, VMError> {
        let top = self.spine.pop().expect("FIXME");
        let mut next = top.unpack_arg();

        // Prepare spine for continuation
        match io {
            IO::Bind { cc } => {
                let Some(app) = self.spine.pop().ok() else {
                    runtime_crash!("Could not pop arg 1 while setting up IO {io}")
                };
                let continuation = app.unpack_arg();
                if cc {
                    let Some(app) = self.spine.pop().ok() else {
                        runtime_crash!("Could not pop arg 1 while setting up IO {io}")
                    };
                    let k = app.unpack_arg();

                    next = Pointer::new(mc, App { fun: next, arg: k });
                }
                self.spine.push(continuation);
            }
            IO::Then => {
                let Some(app) = self.spine.pop().ok() else {
                    runtime_crash!("Could not pop arg 1 while setting up IO {io}")
                };
                let continuation = app.unpack_arg();
                self.spine.push(continuation);
            }
            IO::Perform => {
                // Push the top app node back onto the spine. We will overwrite it later.
                self.spine.push(top);
            }
        }

        self.io_context.push(io);
        Ok(next)
    }

    fn resume_io(
        &mut self,
        mc: &Mutation<'gc>,
        value: Pointer<'gc>,
    ) -> Result<Pointer<'gc>, VMError> {
        let io = self.io_context.pop().ok_or(VMError::AlreadyDone)?;
        let continuation = self
            .spine
            .pop()
            .map_err(|source| VMError::ResumeIO { io, source })?;

        let next = match io {
            IO::Bind { .. } => {
                let p = Pointer::new(
                    mc,
                    App {
                        fun: continuation,
                        arg: value,
                    },
                );
                debug!("    >>= {continuation:?}");
                debug!("    ... {value:?}");
                debug!("     =  made {p:?}");
                p
            }

            IO::Then => continuation, // Ignore the monadic value, return the continuation
            IO::Perform => {
                // Set the top node applying PerformIO to the resulting value
                continuation.set(mc, Value::Ref(value));
                value // Return the "unwrapped" monadic value
            }
        };
        Ok(next)
    }

    fn handle_unop<P, R>(
        &mut self,
        mc: &Mutation<'gc>,
        op: impl Display + Reduce,
        handler: impl FnOnce(P) -> R,
    ) -> Result<Pointer<'gc>, VMError>
    where
        P: Copy + FromValue<'gc>,
        R: IntoValue<'gc>,
    {
        debug!("handling unop {op}");
        let mut top = self.spine.pop().expect("FIXME");
        top.forward(mc);
        debug!("    arg 1: {:?}", top.unpack_arg());
        let arg1 = top.unpack_arg().follow().unpack();
        let arg1 = P::from_value(mc, arg1).expect("FIXME");

        let value = handler(arg1).into_value(mc);
        debug!("    returned: {value:?}");

        if op.io_action() {
            self.resume_io(mc, Pointer::new(mc, value))
        } else {
            top.set(mc, value);
            Ok(top)
        }
    }

    fn handle_binop<P, Q, R>(
        &mut self,
        mc: &Mutation<'gc>,
        op: impl Display + Reduce,
        handler: impl FnOnce(P, Q) -> R,
    ) -> Result<Pointer<'gc>, VMError>
    where
        P: Copy + FromValue<'gc>,
        Q: Copy + FromValue<'gc>,
        R: IntoValue<'gc>,
    {
        debug!("handling binop {op}");
        let mut top = self.spine.pop().expect("FIXME");
        top.forward(mc);
        debug!("    arg 1: {:?}", top.unpack_arg());
        let arg1 = top.unpack_arg().follow().unpack();
        let arg1 = P::from_value(mc, arg1).unwrap();

        let mut top = self.spine.pop().expect("FIXME");
        top.forward(mc);
        debug!("    arg 2: {:?}", top.unpack_arg());
        let arg2 = top.unpack_arg().follow().unpack();
        let arg2 = Q::from_value(mc, arg2).unwrap();

        let value = handler(arg1, arg2).into_value(mc);
        debug!("    returned: {value:?}");

        if op.io_action() {
            self.resume_io(mc, Pointer::new(mc, value))
        } else {
            top.set(mc, value);
            Ok(top)
        }
    }

    /// `next` has been reduced to the given combinator, with all strict arguments evaluated.
    fn eval_comb(&mut self, comb: Combinator, mc: &Mutation<'gc>) -> Result<Pointer<'gc>, VMError> {
        match comb {
            // The Turner combinators can all be handled by their redux codes
            Combinator::S
            | Combinator::K
            | Combinator::I
            | Combinator::B
            | Combinator::C
            | Combinator::A
            | Combinator::Y
            | Combinator::SS
            | Combinator::BB
            | Combinator::CC
            | Combinator::P
            | Combinator::R
            | Combinator::O
            | Combinator::U
            | Combinator::Z
            | Combinator::K2
            | Combinator::K3
            | Combinator::K4
            | Combinator::CCB
            | Combinator::Seq => self.do_redux(mc, comb),

            Combinator::Rnf => unimplemented!("RNF not implemented"),
            Combinator::ErrorMsg => todo!("{comb:?}"),
            Combinator::NoDefault => todo!("{comb:?}"),
            Combinator::NoMatch => todo!("{comb:?}"),

            Combinator::Equal => todo!("{comb:?}"),
            Combinator::StrEqual => todo!("{comb:?}"),
            Combinator::Compare => todo!("{comb:?}"),
            Combinator::StrCmp => todo!("{comb:?}"),
            Combinator::IntCmp => self.handle_binop(mc, comb, |x: Integer, y: Integer| x.cmp(&y)),
            Combinator::ToInt => todo!("{comb:?}"),
            Combinator::ToFloat => todo!("{comb:?}"),

            // Characters and integers have the same underlying representation
            Combinator::Ord => self.handle_unop(mc, comb, |i: Integer| i),
            // Characters and integers have the same underlying representation
            Combinator::Chr => self.handle_unop(mc, comb, |i: Integer| i),

            Combinator::Neg => self.handle_unop(mc, comb, Integer::ineg),
            Combinator::Add => self.handle_binop(mc, comb, Integer::iadd),
            Combinator::Sub => self.handle_binop(mc, comb, Integer::isub),
            Combinator::Subtract => self.handle_binop(mc, comb, Integer::subtract),
            Combinator::Mul => self.handle_binop(mc, comb, Integer::imul),
            Combinator::Quot => self.handle_binop(mc, comb, Integer::iquot),
            Combinator::Rem => self.handle_binop(mc, comb, Integer::irem),
            Combinator::Eq => self.handle_binop(mc, comb, Integer::ieq),
            Combinator::Ne => self.handle_binop(mc, comb, Integer::ine),
            Combinator::Lt => self.handle_binop(mc, comb, Integer::ilt),
            Combinator::Le => self.handle_binop(mc, comb, Integer::ile),
            Combinator::Gt => self.handle_binop(mc, comb, Integer::igt),
            Combinator::Ge => self.handle_binop(mc, comb, Integer::ige),
            Combinator::UQuot => self.handle_binop(mc, comb, Integer::uquot),
            Combinator::URem => self.handle_binop(mc, comb, Integer::urem),
            Combinator::ULt => self.handle_binop(mc, comb, Integer::ult),
            Combinator::ULe => self.handle_binop(mc, comb, Integer::ule),
            Combinator::UGt => self.handle_binop(mc, comb, Integer::ugt),
            Combinator::UGe => self.handle_binop(mc, comb, Integer::uge),
            Combinator::Inv => self.handle_unop(mc, comb, Integer::binv),
            Combinator::And => self.handle_binop(mc, comb, Integer::band),
            Combinator::Or => self.handle_binop(mc, comb, Integer::bor),
            Combinator::Xor => self.handle_binop(mc, comb, Integer::bxor),
            Combinator::Shl => self.handle_binop(mc, comb, Integer::ushl),
            Combinator::Shr => self.handle_binop(mc, comb, Integer::ushr),
            Combinator::AShr => self.handle_binop(mc, comb, Integer::ashr),

            Combinator::FNeg => self.handle_unop(mc, comb, Float::fneg),
            Combinator::FAdd => self.handle_binop(mc, comb, Float::fadd),
            Combinator::FSub => self.handle_binop(mc, comb, Float::fsub),
            Combinator::FMul => self.handle_binop(mc, comb, Float::fmul),
            Combinator::FDiv => self.handle_binop(mc, comb, Float::fdiv),
            Combinator::FEq => self.handle_binop(mc, comb, Float::feq),
            Combinator::FNe => self.handle_binop(mc, comb, Float::fne),
            Combinator::FLt => self.handle_binop(mc, comb, Float::flt),
            Combinator::FLe => self.handle_binop(mc, comb, Float::fle),
            Combinator::FGt => self.handle_binop(mc, comb, Float::fgt),
            Combinator::FGe => self.handle_binop(mc, comb, Float::fge),
            Combinator::IToF => self.handle_unop(mc, comb, Float::from_integer),
            Combinator::FShow => self.handle_unop(mc, comb, FShow),
            Combinator::FRead => todo!("{comb:?}"),

            Combinator::PNull => {
                unreachable!("0-arity combinator {comb} should not exist at runtime")
            }
            Combinator::ToPtr => todo!("{comb:?}"),
            Combinator::PCast => self.handle_unop(mc, comb, |i: Value<'_>| i),
            Combinator::PEq => self.handle_binop(mc, comb, ForeignPtr::peq),
            Combinator::PAdd => self.handle_binop(mc, comb, ForeignPtr::padd),
            Combinator::PSub => self.handle_binop(mc, comb, ForeignPtr::psub),

            Combinator::Return => self.handle_unop(mc, comb, |v: Value<'_>| v),
            Combinator::Bind => self.start_io(mc, IO::Bind { cc: false }),
            Combinator::CCBind => self.start_io(mc, IO::Bind { cc: true }),
            Combinator::Then => self.start_io(mc, IO::Then),
            Combinator::PerformIO => self.start_io(mc, IO::Perform),

            Combinator::Catch => todo!("{comb:?}: need to save stack state"),
            Combinator::DynSym => todo!("{comb:?}: dynamic FFI"),

            Combinator::Serialize => unimplemented!("serialization not supported"),
            Combinator::Deserialize => unimplemented!("deserialization not supported"),

            Combinator::StdIn | Combinator::StdOut | Combinator::StdErr => {
                unreachable!("0-arity combinator {comb} should not exist at runtime")
            }
            Combinator::Print => todo!("{comb:?}"),
            Combinator::GetArgRef => todo!("{comb:?}"),
            Combinator::GetTimeMilli => todo!("{comb:?}"),

            Combinator::ArrAlloc => todo!("{comb:?}"),
            Combinator::ArrSize => todo!("{comb:?}"),
            Combinator::ArrRead => todo!("{comb:?}"),
            Combinator::ArrWrite => todo!("{comb:?}"),
            Combinator::ArrEq => todo!("{comb:?}"),

            Combinator::FromUtf8 => {
                let top = self.spine.pop().unwrap();
                let s = top.unpack_arg().unpack().unwrap_string();
                let res = crate::string::hstring_from_utf8(mc, s.as_ref());
                let res = Pointer::new(mc, res);
                top.set(mc, res);
                Ok(res)
                // TODO: make sure this is actually calling hstring_from_utf8
                // self.handle_unop(mc, comb, Value::unwrap_string)
            }
            Combinator::CAStringLenNew => {
                let top = self.spine.pop().expect("FIXME");
                let s = top.unpack_arg();
                if let Some(cas) = new_castringlen(mc, s) {
                    self.resume_io(mc, cas)
                } else {
                    let new = Pointer::new(mc, Combinator::CAStringLenNew);
                    let again = Pointer::new(mc, App { fun: new, arg: s });
                    top.set(mc, eval_hstring(mc, s, again));
                    Ok(top)
                }
            }
            Combinator::CAStringPeek => self.handle_unop(mc, comb, peek_castring),
            Combinator::CAStringPeekLen => self.handle_binop(mc, comb, peek_castringlen),
        }
    }

    fn eval_ffi(&mut self, ffi: FfiSymbol, mc: &Mutation<'gc>) -> Result<Pointer<'gc>, VMError> {
        macro_rules! ffi {
            ($func:path:(_)) => {
                |x: _| unsafe { $func(x) }
            };
            ($func:path:(_, _)) => {
                |x: _, y: _| unsafe { $func(x, y) }
            };
        }

        // FFI calls are made in the IO monad, so we need to handle it as such
        match ffi {
            FfiSymbol::acos => self.handle_unop(mc, ffi, Float::facos),
            FfiSymbol::asin => self.handle_unop(mc, ffi, Float::fasin),
            FfiSymbol::atan => self.handle_unop(mc, ffi, Float::fatan),
            FfiSymbol::atan2 => self.handle_binop(mc, ffi, Float::fatan2),
            FfiSymbol::cos => self.handle_unop(mc, ffi, Float::fcos),
            FfiSymbol::exp => self.handle_unop(mc, ffi, Float::fexp),
            FfiSymbol::log => self.handle_unop(mc, ffi, Float::flog),
            FfiSymbol::sin => self.handle_unop(mc, ffi, Float::fsin),
            FfiSymbol::sqrt => self.handle_unop(mc, ffi, Float::fsqrt),
            FfiSymbol::tan => self.handle_unop(mc, ffi, Float::ftan),
            FfiSymbol::fopen => self.handle_binop(mc, ffi, ffi!(libc::fopen:(_, _))),
            FfiSymbol::add_FILE => self.handle_unop(mc, ffi, ffi!(ffi::bfile::add_FILE:(_))),
            FfiSymbol::add_utf8 => self.handle_unop(mc, ffi, ffi!(ffi::bfile::add_utf8:(_))),
            FfiSymbol::closeb => self.handle_unop(mc, ffi, ffi!(ffi::bfile::closeb:(_))),
            FfiSymbol::flushb => self.handle_unop(mc, ffi, ffi!(ffi::bfile::flushb:(_))),
            FfiSymbol::getb => self.handle_unop(mc, ffi, ffi!(ffi::bfile::getb:(_))),
            FfiSymbol::putb => self.handle_binop(mc, ffi, ffi!(ffi::bfile::putb:(_, _))),
            FfiSymbol::ungetb => self.handle_binop(mc, ffi, ffi!(ffi::bfile::ungetb:(_, _))),
            FfiSymbol::system => self.handle_unop(mc, ffi, ffi!(libc::system:(_))),
            FfiSymbol::tmpname => todo!("{ffi:?}"),
            FfiSymbol::unlink => self.handle_unop(mc, ffi, ffi!(libc::unlink:(_))),
            FfiSymbol::malloc => self.handle_unop(mc, ffi, ffi!(libc::malloc:(_))),
            FfiSymbol::free => self.handle_unop(mc, ffi, ffi!(libc::free:(_))),
        }
    }

    fn handle_tick(&mut self, t: Tick) -> Result<Pointer<'gc>, VMError> {
        self.tick_table.tick(t)?;
        let next = self
            .spine
            .pop()
            .map_err(|source| VMError::TickResume {
                tick: self.tick_table.info(t).clone(),
                source,
            })?
            .unpack_arg();
        Ok(next)
    }

    fn step(&mut self, mc: &Mutation<'gc>) -> Result<(), VMError> {
        debug!("BEGIN step");

        let mut cur = self.tip;
        cur.forward(mc);

        debug!("handling node: {cur:?}");
        self.tip = match cur.unpack() {
            Value::Ref(_) => {
                unreachable!("indirections should already have been followed")
            }
            Value::App(App { fun, arg: _arg }) => {
                debug!("Pushed {fun:?} to spine");
                self.spine.push(cur);
                Ok(fun)
            }
            Value::Combinator(comb) => self.start_strict(comb.into(), mc),
            Value::String(_) | Value::Integer(_) | Value::Float(_) | Value::ForeignPtr(_) => {
                self.resume_strict(mc)
            }
            Value::Tick(t) => self.handle_tick(t),
            Value::Ffi(ffi) => self.start_strict(ffi.into(), mc),
            Value::BadDyn(sym) => Err(VMError::BadDyn(sym.name().to_string())),
        }?;

        debug!("END step");
        Ok(())
    }
}

pub struct VM {
    arena: Arena<Rootable![State<'_>]>,
}

impl VM {
    pub fn step(&mut self) -> Result<(), VMError> {
        self.arena.mutate_root(|mc, st| st.step(mc))
    }
}

impl Debug for VM {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // TODO: provide stats here?
        f.debug_struct("VM").finish_non_exhaustive()
    }
}

#[cfg(feature = "file")]
impl From<crate::file::Program> for VM {
    fn from(value: crate::file::Program) -> Self {
        Self {
            arena: Arena::new(|mc| {
                let mut tbl = TickTable::new();
                let top = value.deserialize(mc, &mut tbl);
                State::new(top, tbl)
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gc_arena::Arena;

    macro_rules! comb {
        ($mc:ident, $c:ident) => {
            Pointer::new($mc, Value::Combinator(Combinator::$c))
        };
    }
    macro_rules! int {
        ($mc:ident, $i:literal) => {
            Pointer::new($mc, Value::Integer(Integer::from($i)))
        };
    }
    macro_rules! app {
        // Coercions, using other macros and threading through mc
        ($mc:ident, #$i:literal $($rest:tt)*) => {
            app!($mc, int!($mc, $i) $($rest)*)
        };
        ($mc:ident, $f:expr, #$i:literal $($rest:tt)*) => {
            app!($mc, $f, int!($mc, $i) $($rest)*)
        };
        ($mc:ident, :$comb:tt $($rest:tt)*) => {
            app!($mc, comb!($mc, $comb) $($rest)*)
        };
        ($mc:ident, $f:expr, :$comb:tt $($rest:tt)*) => {
            app!($mc, $f, comb!($mc, $comb) $($rest)*)
        };
        ($mc:ident, @($($app:tt)*) $($rest:tt)*) => {
            app!($mc, app!($mc, $($app)*) $($rest)*)
        };
        ($mc:ident, $f:expr, @($($app:tt)*) $($rest:tt)*) => {
            app!($mc, $f, app!($mc, $($app)*) $($rest)*)
        };

        // Base case
        ($mc:ident, $f:expr, $a:expr) => {
            Pointer::new($mc, Value::App(App { fun: $f, arg: $a }))
        };
        ($mc:ident, $f:expr, $a:expr, $($rest:tt)+) => {
            app!($mc, Pointer::new($mc, Value::App(App { fun: $f, arg: $a })), $($rest)+)
        };
    }

    #[test]
    fn arith420() {
        type Root = Rootable![(State<'_>, Pointer<'_>)];
        let mut arena: Arena<Root> = Arena::new(|mc| {
            let top = app!(mc, :Mul, #42, #10);
            (State::new(top, Default::default()), top)
        }); // tip points to ((* 42) 10)
        arena.mutate_root(|mc, (st, _)| st.step(mc)).unwrap(); // tip points to (* 42)
        arena.mutate_root(|mc, (st, _)| st.step(mc)).unwrap(); // tip points to *
        arena.mutate_root(|mc, (st, _)| st.step(mc)).unwrap(); // reduce *, tip points to result (420)
        arena.mutate(|_, (st, top)| {
            assert_eq!(st.tip, *top, "tip is back at top");
            assert_eq!(
                top.unpack(),
                Value::Integer(Integer::from(420)),
                "top was modified in place, to value of 420",
            );
        });
    }

    #[test]
    fn arith420_nested() {
        // TODO: something like (40 + 2) * 10
    }

    #[test]
    fn skkr() {
        type Root = Rootable![(State<'_>, Pointer<'_>)];
        let mut arena: Arena<Root> = Arena::new(|mc| {
            let (s, k, r) = (comb!(mc, S), comb!(mc, K), comb!(mc, R));
            let top = app!(mc, s, k, k, r);
            (State::new(top, Default::default()), top)
        }); // tip points to (((S K) K) R)
        arena.mutate_root(|mc, (st, _)| st.step(mc)).unwrap(); // tip points to ((S K) K)
        arena.mutate_root(|mc, (st, _)| st.step(mc)).unwrap(); // tip points to (S K)
        arena.mutate_root(|mc, (st, _)| st.step(mc)).unwrap(); // tip points to S
        arena.mutate_root(|mc, (st, _)| st.step(mc)).unwrap(); // reduce S, tip points to ((K R) (K R))
        arena.mutate_root(|mc, (st, _)| st.step(mc)).unwrap(); // tip points to (K R)
        arena.mutate_root(|mc, (st, _)| st.step(mc)).unwrap(); // tip points to K
        arena.mutate_root(|mc, (st, _)| st.step(mc)).unwrap(); // reduce K, tip points to *->R

        arena.mutate(|_, (st, top)| {
            assert_eq!(
                st.tip.follow(),
                top.follow(),
                "tip is back at top (modulo indirection)"
            );
            assert!(
                matches!(top.unpack(), Value::Ref(_)),
                "top should be indirection"
            );
            let top = top.follow();
            assert_eq!(top.unpack(), Value::Combinator(Combinator::R));
        });
    }

    #[test]
    fn io_traversal() {
        type Root = Rootable![(State<'_>, Pointer<'_>)];
        let mut arena: Arena<Root> = Arena::new(|mc| {
            let bottom = app!(mc, :Neg, :Neg); // Should never be eval'd
            let result = int!(mc, 42);
            let inner = app!(mc, :Then, @(:Return, bottom), @(:Return, result));
            let top = app!(mc, :Bind, inner, :Return);
            (State::new(top, Default::default()), result)
        });
        // tip points to ((>>= ((>> (return_0 bottom)) (return_1 result))) return_2)
        arena.mutate_root(|mc, (st, _)| st.step(mc)).unwrap(); // tip points to (>>= ((>> (return_0 bottom)) (return_1 result)))
        arena.mutate_root(|mc, (st, _)| st.step(mc)).unwrap(); // tip points to >>=
        arena.mutate_root(|mc, (st, _)| st.step(mc)).unwrap(); // reduce >>=, tip points to ((>> (return_0 bottom)) (return_1 result))
        arena.mutate_root(|mc, (st, _)| st.step(mc)).unwrap(); // tip points to (>> (return_0 bottom))
        arena.mutate_root(|mc, (st, _)| st.step(mc)).unwrap(); // tip points to >>
        arena.mutate_root(|mc, (st, _)| st.step(mc)).unwrap(); // reduces >>, tip points to (return_1 result)
        arena.mutate_root(|mc, (st, _)| st.step(mc)).unwrap(); // tip points to return_1
        arena.mutate_root(|mc, (st, _)| st.step(mc)).unwrap(); // reduce return_1, tip points to (return_2 result)

        arena.mutate(|_, (st, result)| {
            let tip = st.tip;
            assert!(
                matches!(tip.unpack(), Value::App(_)),
                "top should be application, but it was {:#?}",
                tip.unpack()
            );
            assert_eq!(
                tip.unpack_fun().unpack(),
                Value::Combinator(Combinator::Return)
            );
            assert_eq!(tip.unpack_arg(), *result);
        });
    }

    #[test]
    fn seq() {
        // TODO: use seq to check that strictness is working as expected
    }
}
