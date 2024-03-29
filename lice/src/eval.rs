#![allow(dead_code)]
use crate::{
    combinator::{Combinator, Reduce, ReduxCode},
    ffi::{self, FfiSymbol},
    float::Float,
    integer::Integer,
    memory::{App, Pointer, Value},
    tick::TickTable,
};
use bitvec::prelude::*;
use core::{
    cell::OnceCell,
    fmt::{self, Debug, Display},
};
use gc_arena::{Arena, Collect, Mutation, Rootable};
use log::{debug, info};

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

#[derive(Collect, Default)]
#[collect(no_drop)]
struct Spine<'gc> {
    stack: Vec<Pointer<'gc>>,
}

impl<'gc> Spine<'gc> {
    fn new() -> Self {
        Self::default()
    }

    fn push(&mut self, app: Pointer<'gc>) {
        self.stack.push(app);
    }

    fn pop(&mut self) -> Option<Pointer<'gc>> {
        self.stack.pop()
    }

    fn peek(&self, idx: usize) -> Option<Pointer<'gc>> {
        let idx = self.stack.len() - 1 - idx;
        let app = self.stack.get(idx)?;
        Some(*app)
    }
}

#[derive(Debug, Collect, Clone, Copy)]
#[collect(require_static)]
enum Strict {
    Combinator(Combinator),
    Ffi(FfiSymbol),
}

impl Strict {
    fn strictness(&self) -> &'static [bool] {
        match self {
            Strict::Combinator(c) => c.strictness(),
            Strict::Ffi(f) => f.strictness(),
        }
    }
}

/// Pending strict evaluation some combinator's arguments.
#[derive(Debug, Collect, Clone, Copy)]
#[collect(require_static)]
struct StrictWork {
    strict: Strict,
    args: BitArr!(for Self::MAX_ARGS),
}

impl StrictWork {
    const MAX_ARGS: usize = 4;

    fn init<'gc>(strict: Strict, spine: &Spine<'gc>) -> Option<(Self, Pointer<'gc>)> {
        // Figure out which args need to be evaluated, i.e., are not yet WHNF
        // Default state: we don't need to evalute each arg
        let mut args = BitArray::new([0]);

        // Also find index of first argument that needs evaluating
        let first = OnceCell::new();

        for (idx, is_strict) in strict.strictness().iter().enumerate() {
            let Some(app) = spine.peek(idx) else {
                runtime_crash!("Arg {idx} not present for {is_strict:?}")
            };
            let app = app.follow();
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
            Self { strict, args },
            spine.peek(first).unwrap().follow().unpack_arg().follow(),
        ))
    }

    /// Advance the pending arg state.
    fn advance<'gc>(&mut self, spine: &Spine<'gc>) -> Option<Pointer<'gc>> {
        // No need to iterate over index 0 because it should never have been set.
        for idx in 1..Self::MAX_ARGS {
            debug!("advance {:?}: checking arg {idx}", self.strict);
            if self.args[idx] {
                self.args.set(idx, false);

                let Some(app) = spine.peek(idx) else {
                    runtime_crash!(
                        "Arg {idx} not present for strict combinator {:?}",
                        self.strict
                    )
                };
                debug!(
                    "advance {:?}: checking arg: {:?}",
                    self.strict,
                    app.unpack_arg()
                );
                let app = app.follow();
                let arg = app.unpack_arg().follow();
                if !arg.unpack().whnf() {
                    // Only evaluate this arg next if it is not in WHNF yet
                    return Some(app.unpack_arg().follow());
                }
            }
        }
        None
    }

    /// All operations are complete, i.e., nothing is pending.
    fn complete(&self) -> bool {
        self.args.iter().all(|pending| !pending)
    }
}

#[derive(Clone, Collect)]
#[collect(require_static)]
enum IO {
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
    fn do_redux(&mut self, mc: &Mutation<'gc>, comb: Combinator) -> Pointer<'gc> {
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

        for i in 0..comb.arity() {
            let Some(app) = self.spine.pop() else {
                runtime_crash!("Could not pop arg {i} for combinator {comb}")
            };
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
                    unwrap!(stk.push(Pointer::new(mc, Value::App(App { fun, arg }))))
                }
            }
        }

        // We need to handle the last argument differently
        match code[code.len() - 1] {
            ReduxCode::Arg(i) => {
                top.set(mc, Value::Ref(args[i]));
                args[i]
            }
            ReduxCode::Top => {
                unreachable!("no rule should reduce to '^' alone ")
            }
            ReduxCode::App => {
                let arg = unwrap!(stk.pop());
                let fun = unwrap!(stk.pop());
                top.set(mc, Value::App(App { fun, arg }));
                top
            }
        }
    }

    fn start_strict(&mut self, strict: Strict) -> Option<Pointer<'gc>> {
        let (work, first) = StrictWork::init(strict, &self.spine)?;
        self.strict_context.push(work);
        Some(first)
    }

    fn resume_strict(&mut self) -> Result<Pointer<'gc>, Strict> {
        let Some(work) = self.strict_context.last_mut() else {
            runtime_crash!("No pending strict operation");
        };

        if let Some(next) = work.advance(&self.spine) {
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
            let comb = work.strict;
            self.strict_context.pop();
            Err(comb)
        }
    }

    fn start_io(&mut self, mc: &Mutation<'gc>, io: IO) -> Pointer<'gc> {
        let Some(app) = self.spine.pop() else {
            runtime_crash!("Could not pop arg 0 while setting up IO {io}")
        };

        let mut next = app.unpack_arg();

        // Prepare spine for continuation
        match io {
            IO::Bind { cc } => {
                let Some(app) = self.spine.pop() else {
                    runtime_crash!("Could not pop arg 1 while setting up IO {io}")
                };
                let continuation = app.unpack_arg();
                if cc {
                    let Some(app) = self.spine.pop() else {
                        runtime_crash!("Could not pop arg 1 while setting up IO {io}")
                    };
                    let k = app.unpack_arg();

                    next = Pointer::new(mc, Value::App(App { fun: next, arg: k }));
                }
                self.spine.push(continuation);
            }
            IO::Then => {
                let Some(app) = self.spine.pop() else {
                    runtime_crash!("Could not pop arg 1 while setting up IO {io}")
                };
                let continuation = app.unpack_arg();
                self.spine.push(continuation);
            }
            IO::Perform => {
                // Push the app node back onto the spine. We will overwrite it later.
                self.spine.push(app);
            }
        }

        self.io_context.push(io);
        next
    }

    fn resume_io(&mut self, mc: &Mutation<'gc>, value: Pointer<'gc>) -> Pointer<'gc> {
        let Some(io) = self.io_context.pop() else {
            info!("End of program, returned with value: {:?}", value);
            std::process::exit(0);
            // runtime_crash!("No pending strict operation");
        };

        let Some(continuation) = self.spine.pop() else {
            runtime_crash!("Could not pop continuation from spine while resuming {io}")
        };
        match io {
            IO::Bind { .. } => Pointer::new(
                mc,
                // Apply the monadic value to the continuation, and return that
                Value::App(App {
                    fun: continuation,
                    arg: value,
                }),
            ),

            IO::Then => continuation, // Ignore the monadic value, return the continuation
            IO::Perform => {
                // Set the top node applying PerformIO to the resulting value
                // TODO: double check that this is right
                continuation.set(mc, Value::Ref(value));
                value // Return the "unwrapped" monadic value
            }
        }
    }

    fn handle_from_utf8(&mut self, mc: &Mutation<'gc>) -> Pointer<'gc> {
        let c = Combinator::FromUtf8;
        let Some(top) = self.spine.pop() else {
            runtime_crash!("Could not pop arg 0 while performing {c}",)
        };
        let arg = top.unpack_arg();
        let Value::String(arg) = arg.unpack() else {
            runtime_crash!("{c} expected arg to be string, instead got {arg:?}",)
        };

        let arg: &str = arg.as_ref();

        // TODO: don't allocate new combinators
        let mut res = Pointer::new(mc, Value::Combinator(Combinator::NIL));
        if arg.is_empty() {
            return res;
        }
        let cons = Pointer::new(mc, Value::Combinator(Combinator::CONS));

        for c in arg.chars().rev() {
            let data = Pointer::new(mc, Value::Integer(Integer::from(c)));
            let data = Pointer::new(
                mc,
                Value::App(App {
                    fun: cons,
                    arg: data,
                }),
            );
            res = Pointer::new(
                mc,
                Value::App(App {
                    fun: data,
                    arg: res,
                }),
            );
        }

        res
    }

    fn handle_unop<Q, R>(
        &mut self,
        mc: &Mutation<'gc>,
        op: impl Display,
        io: bool,
        handler: impl FnOnce(Q) -> R,
    ) -> Pointer<'gc>
    where
        Q: Copy + TryFrom<Value<'gc>>,
        <Q as TryFrom<Value<'gc>>>::Error: core::fmt::Debug,
        R: Into<Value<'gc>>,
    {
        debug!("handling unop {op}");
        let Some(mut top) = self.spine.pop() else {
            runtime_crash!("Could not pop arg 0 while evaluating operator {op}")
        };
        top.forward(mc);
        debug!("    arg 1: {:?}", top.unpack_arg());
        let arg1 = top.unpack_arg().follow().unpack().try_into().unwrap();

        let value = handler(arg1).into();
        debug!("    returned: {value:?}");

        if io {
            self.resume_io(mc, Pointer::new(mc, value))
        } else {
            top.set(mc, value);
            top
        }
    }

    fn handle_binop<P, Q, R>(
        &mut self,
        mc: &Mutation<'gc>,
        op: impl Display,
        io: bool,
        handler: impl FnOnce(P, Q) -> R,
    ) -> Pointer<'gc>
    where
        P: Copy + TryFrom<Value<'gc>>,
        <P as TryFrom<Value<'gc>>>::Error: core::fmt::Debug,
        Q: Copy + TryFrom<Value<'gc>>,
        <Q as TryFrom<Value<'gc>>>::Error: core::fmt::Debug,
        R: Into<Value<'gc>>,
    {
        debug!("handling binop {op}");
        let Some(mut top) = self.spine.pop() else {
            runtime_crash!("Could not pop arg 0 while evaluating operator {op}")
        };
        top.forward(mc);
        debug!("    arg 1: {:?}", top.unpack_arg());
        let arg1 = top.unpack_arg().follow().unpack().try_into().unwrap();
        let Some(mut top) = self.spine.pop() else {
            runtime_crash!("Could not pop arg 1 while evaluating operator {op}")
        };
        top.forward(mc);
        debug!("    arg 2: {:?}", top.unpack_arg());
        let arg2 = top.unpack_arg().follow().unpack().try_into().unwrap();

        let value = handler(arg1, arg2).into();
        debug!("    returned: {value:?}");

        if io {
            self.resume_io(mc, Pointer::new(mc, value))
        } else {
            top.set(mc, value);
            top
        }
    }

    /// `next` has been reduced to the given combinator, with all strict arguments evaluated.
    fn handle_comb(&mut self, comb: Combinator, mc: &Mutation<'gc>) -> Pointer<'gc> {
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
            | Combinator::CCB => self.do_redux(mc, comb),

            Combinator::Seq => self.do_redux(mc, comb), // Seq has a trivial redux code
            Combinator::Rnf => unimplemented!("RNF not implemented"),
            Combinator::ErrorMsg => todo!("{comb:?}"),
            Combinator::NoDefault => todo!("{comb:?}"),
            Combinator::NoMatch => todo!("{comb:?}"),

            Combinator::Equal => todo!("{comb:?}"),
            Combinator::StrEqual => todo!("{comb:?}"),
            Combinator::Compare => todo!("{comb:?}"),
            Combinator::StrCmp => todo!("{comb:?}"),
            Combinator::IntCmp => todo!("{comb:?}"),
            Combinator::ToInt => todo!("{comb:?}"),
            Combinator::ToFloat => todo!("{comb:?}"),

            // Characters and integers have the same underlying representation
            Combinator::Ord => self.handle_unop(mc, comb, false, |i: Integer| i),
            // Characters and integers have the same underlying representation
            Combinator::Chr => self.handle_unop(mc, comb, false, |i: Integer| i),

            Combinator::Neg => self.handle_unop(mc, comb, false, Integer::ineg),
            Combinator::Add => self.handle_binop(mc, comb, false, Integer::iadd),
            Combinator::Sub => self.handle_binop(mc, comb, false, Integer::isub),
            Combinator::Subtract => self.handle_binop(mc, comb, false, Integer::subtract),
            Combinator::Mul => self.handle_binop(mc, comb, false, Integer::imul),
            Combinator::Quot => self.handle_binop(mc, comb, false, Integer::iquot),
            Combinator::Rem => self.handle_binop(mc, comb, false, Integer::irem),
            Combinator::Eq => self.handle_binop(mc, comb, false, Integer::ieq),
            Combinator::Ne => self.handle_binop(mc, comb, false, Integer::ine),
            Combinator::Lt => self.handle_binop(mc, comb, false, Integer::ilt),
            Combinator::Le => self.handle_binop(mc, comb, false, Integer::ile),
            Combinator::Gt => self.handle_binop(mc, comb, false, Integer::igt),
            Combinator::Ge => self.handle_binop(mc, comb, false, Integer::ige),
            Combinator::UQuot => self.handle_binop(mc, comb, false, Integer::uquot),
            Combinator::URem => self.handle_binop(mc, comb, false, Integer::urem),
            Combinator::ULt => self.handle_binop(mc, comb, false, Integer::ult),
            Combinator::ULe => self.handle_binop(mc, comb, false, Integer::ule),
            Combinator::UGt => self.handle_binop(mc, comb, false, Integer::ugt),
            Combinator::UGe => self.handle_binop(mc, comb, false, Integer::uge),
            Combinator::Inv => self.handle_unop(mc, comb, false, Integer::binv),
            Combinator::And => self.handle_binop(mc, comb, false, Integer::band),
            Combinator::Or => self.handle_binop(mc, comb, false, Integer::bor),
            Combinator::Xor => self.handle_binop(mc, comb, false, Integer::bxor),
            Combinator::Shl => self.handle_binop(mc, comb, false, Integer::ushl),
            Combinator::Shr => self.handle_binop(mc, comb, false, Integer::ushr),
            Combinator::AShr => self.handle_binop(mc, comb, false, Integer::ashr),

            Combinator::FNeg => self.handle_unop(mc, comb, false, Float::fneg),
            Combinator::FAdd => self.handle_binop(mc, comb, false, Float::fadd),
            Combinator::FSub => self.handle_binop(mc, comb, false, Float::fsub),
            Combinator::FMul => self.handle_binop(mc, comb, false, Float::fmul),
            Combinator::FDiv => self.handle_binop(mc, comb, false, Float::fdiv),
            Combinator::FEq => self.handle_binop(mc, comb, false, Float::feq),
            Combinator::FNe => self.handle_binop(mc, comb, false, Float::fne),
            Combinator::FLt => self.handle_binop(mc, comb, false, Float::flt),
            Combinator::FLe => self.handle_binop(mc, comb, false, Float::fle),
            Combinator::FGt => self.handle_binop(mc, comb, false, Float::fgt),
            Combinator::FGe => self.handle_binop(mc, comb, false, Float::fge),
            Combinator::IToF => self.handle_unop(mc, comb, false, Float::from_integer),
            Combinator::FShow => todo!("{comb:?}"),
            Combinator::FRead => todo!("{comb:?}"),

            Combinator::PNull => {
                unreachable!("0-arity combinator {comb} should not exist at runtime")
            }
            Combinator::ToPtr => todo!("{comb:?}"),
            Combinator::PCast => self.handle_unop(mc, comb, false, |i: Integer| i),
            Combinator::PEq => todo!("{comb:?}"),
            Combinator::PAdd => todo!("{comb:?}"),
            Combinator::PSub => todo!("{comb:?}"),

            Combinator::Return => {
                let Some(top) = self.spine.pop() else {
                    runtime_crash!("Could not pop arg 0 while evaluating operator {comb}")
                };
                let value = top.unpack_arg();
                self.resume_io(mc, value)
            }
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

            Combinator::FromUtf8 => self.handle_from_utf8(mc),
            Combinator::CStringNew => todo!("{comb:?}"),
            Combinator::CStringPeek => todo!("{comb:?}"),
            Combinator::CStringPeekLen => todo!("{comb:?}"),
        }
    }

    fn handle_ffi(&mut self, ffi: FfiSymbol, mc: &Mutation<'gc>) -> Pointer<'gc> {
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
            FfiSymbol::acos => self.handle_unop(mc, ffi, true, Float::facos),
            FfiSymbol::asin => self.handle_unop(mc, ffi, true, Float::fasin),
            FfiSymbol::atan => self.handle_unop(mc, ffi, true, Float::fatan),
            FfiSymbol::atan2 => self.handle_binop(mc, ffi, true, Float::fatan2),
            FfiSymbol::cos => self.handle_unop(mc, ffi, true, Float::fcos),
            FfiSymbol::exp => self.handle_unop(mc, ffi, true, Float::fexp),
            FfiSymbol::log => self.handle_binop(mc, ffi, true, Float::flog),
            FfiSymbol::sin => self.handle_unop(mc, ffi, true, Float::fsin),
            FfiSymbol::sqrt => self.handle_unop(mc, ffi, true, Float::fsqrt),
            FfiSymbol::tan => self.handle_unop(mc, ffi, true, Float::ftan),
            FfiSymbol::fopen => self.handle_binop(mc, ffi, true, ffi!(libc::fopen:(_, _))),
            FfiSymbol::add_FILE => self.handle_unop(mc, ffi, true, ffi!(ffi::bfile::add_FILE:(_))),
            FfiSymbol::add_utf8 => self.handle_unop(mc, ffi, true, ffi!(ffi::bfile::add_utf8:(_))),
            FfiSymbol::closeb => self.handle_unop(mc, ffi, true, ffi!(ffi::bfile::closeb:(_))),
            FfiSymbol::flushb => self.handle_unop(mc, ffi, true, ffi!(ffi::bfile::flushb:(_))),
            FfiSymbol::getb => self.handle_unop(mc, ffi, true, ffi!(ffi::bfile::getb:(_))),
            FfiSymbol::putb => self.handle_binop(mc, ffi, true, ffi!(ffi::bfile::putb:(_, _))),
            FfiSymbol::ungetb => self.handle_binop(mc, ffi, true, ffi!(ffi::bfile::ungetb:(_, _))),
            FfiSymbol::system => self.handle_unop(mc, ffi, true, ffi!(libc::system:(_))),
            FfiSymbol::tmpname => todo!("{ffi:?}"),
            FfiSymbol::unlink => self.handle_unop(mc, ffi, true, ffi!(libc::unlink:(_))),
        }
    }

    fn step(&mut self, mc: &Mutation<'gc>) {
        debug!("BEGIN step");

        let mut cur = self.tip;
        cur.forward(mc);

        debug!("handling node: {cur:?}");
        self.tip = match cur.unpack() {
            Value::Ref(_) => {
                unreachable!("indirections should already have been followed")
            }
            Value::App(App { fun, arg: _arg }) => {
                self.spine.push(cur);
                fun
            }
            Value::Combinator(comb) => self
                // `next` has been reduced to a combinator, though one or more of its arguments
                // may still need to be evaluated before it can be handled
                .start_strict(Strict::Combinator(comb))
                // If start_strict returns None, then we can directly handle the combinator
                .unwrap_or_else(|| self.handle_comb(comb, mc)),
            Value::String(_) | Value::Integer(_) | Value::Float(_) | Value::ForeignPtr(_) => self
                // `next` has been reduced to a value, probably because some strict operator had
                // previously demanded strict arguments. We check our strict continuation context
                // to see if we still need to evaluate more arguments, or if we can now handle the
                // strict operator.
                .resume_strict()
                .unwrap_or_else(|strict| match strict {
                    Strict::Combinator(comb) => self.handle_comb(comb, mc),
                    Strict::Ffi(ffi) => self.handle_ffi(ffi, mc),
                }),
            Value::Tick(t) => {
                self.tick_table.tick(t);
                let Some(next) = self.spine.pop() else {
                    runtime_crash!(
                        "Could not pop arg while evaluating tick {}",
                        self.tick_table.info(t).name
                    )
                };
                next.unpack_arg()
            }
            Value::Ffi(ffi) => self
                .start_strict(Strict::Ffi(ffi))
                .unwrap_or_else(|| self.handle_ffi(ffi, mc)),
            Value::BadDyn(sym) => panic!("missing FFI symbol: {}", sym.name()),
        };

        debug!("END step");
    }
}

pub struct VM {
    arena: Arena<Rootable![State<'_>]>,
}

impl VM {
    pub fn step(&mut self) {
        self.arena.mutate_root(|mc, st| st.step(mc));
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
        arena.mutate_root(|mc, (st, _)| st.step(mc)); // tip points to (* 42)
        arena.mutate_root(|mc, (st, _)| st.step(mc)); // tip points to *
        arena.mutate_root(|mc, (st, _)| st.step(mc)); // reduce *, tip points to result (420)
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
        arena.mutate_root(|mc, (st, _)| st.step(mc)); // tip points to ((S K) K)
        arena.mutate_root(|mc, (st, _)| st.step(mc)); // tip points to (S K)
        arena.mutate_root(|mc, (st, _)| st.step(mc)); // tip points to S
        arena.mutate_root(|mc, (st, _)| st.step(mc)); // reduce S, tip points to ((K R) (K R))
        arena.mutate_root(|mc, (st, _)| st.step(mc)); // tip points to (K R)
        arena.mutate_root(|mc, (st, _)| st.step(mc)); // tip points to K
        arena.mutate_root(|mc, (st, _)| st.step(mc)); // reduce K, tip points to *->R

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
        arena.mutate_root(|mc, (st, _)| st.step(mc)); // tip points to (>>= ((>> (return_0 bottom)) (return_1 result)))
        arena.mutate_root(|mc, (st, _)| st.step(mc)); // tip points to >>=
        arena.mutate_root(|mc, (st, _)| st.step(mc)); // reduce >>=, tip points to ((>> (return_0 bottom)) (return_1 result))
        arena.mutate_root(|mc, (st, _)| st.step(mc)); // tip points to (>> (return_0 bottom))
        arena.mutate_root(|mc, (st, _)| st.step(mc)); // tip points to >>
        arena.mutate_root(|mc, (st, _)| st.step(mc)); // reduces >>, tip points to (return_1 result)
        arena.mutate_root(|mc, (st, _)| st.step(mc)); // tip points to return_1
        arena.mutate_root(|mc, (st, _)| st.step(mc)); // reduce return_1, tip points to (return_2 result)

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
