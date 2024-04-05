use crate::{
    combinator::{Combinator, Reduce, ReduxCode},
    ffi::{self, FfiSymbol, ForeignPtr},
    float::{FShow, Float},
    integer::Integer,
    memory::{App, ConversionError, FromValue, IntoValue, Pointer, Value},
    string::{eval_hstring, new_castringlen, peek_castring, peek_castringlen},
    tick::{Tick, TickError, TickInfo, TickTable},
};
use bitvec::prelude::*;
use core::{cell::OnceCell, fmt::Debug};
use debug_unwraps::DebugUnwrapExt;
use gc_arena::{Arena, Collect, Mutation, Rootable};

/// Shorthand for my tomfoolery
macro_rules! expect {
    ($e:expr $(,)?) => {
        unsafe { ($e).debug_unwrap_unchecked() }
    };
    ($e:expr, $msg:literal $(,)?) => {
        unsafe { ($e).debug_expect_unchecked($msg) }
    };
}

fn unpack_arg(app: Pointer<'_>, index: usize, op: impl Into<Operator>) -> VMResult<Pointer<'_>> {
    let value = app.unpack();
    if let Value::App(App { fun: _, arg }) = value {
        Ok(arg)
    } else {
        Err(VMError::NotApp {
            op: op.into(),
            index,
            got: value.into(),
        })
    }
}

#[derive(Debug, Clone, Copy, derive_more::From, parse_display::Display)]
#[display("Operator({0:?})")]
pub enum Operator {
    Combinator(Combinator),
    FfiSymbol(FfiSymbol),
}

impl Reduce for Operator {
    fn strictness(&self) -> &'static [bool] {
        match self {
            Operator::Combinator(o) => o.strictness(),
            Operator::FfiSymbol(o) => o.strictness(),
        }
    }
    fn redux(&self) -> Option<&'static [ReduxCode]> {
        match self {
            Operator::Combinator(o) => o.redux(),
            Operator::FfiSymbol(o) => o.redux(),
        }
    }
    fn io_action(&self) -> bool {
        match self {
            Operator::Combinator(o) => o.io_action(),
            Operator::FfiSymbol(o) => o.io_action(),
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum SpineError {
    #[error("could not pop, spine empty")]
    NoPop,
    #[error("could not peek, index {0} is out of bounds")]
    NoPeek(usize),
}

type VMResult<T> = Result<T, VMError>;

/// Tidy this up, consolidate some of the errors and consider using `anyhow`.
#[derive(thiserror::Error, Debug)]
pub enum VMError {
    #[error("IO monad terminated with constant value {0}")]
    IOTerminated(Combinator),
    #[error("no IO context to handle {0}")]
    IONoContext(&'static str),
    #[error("could not resume {io}, no continuation found")]
    IONoContinuation { io: Combinator, source: SpineError },
    #[error("could not reduce {op}, expected arg {index} to be an App node but instead got {got}")]
    NotApp {
        op: Operator,
        index: usize,
        got: &'static str,
    },
    #[error("arg {index} of {op} it not the right type")]
    TypeError {
        op: Operator,
        index: usize,
        source: ConversionError,
    },
    #[error("evaluated to unexpected value without strict context: {0}")]
    UnexpectedValue(&'static str),
    #[error(transparent)]
    TickError(#[from] TickError),
    #[error("could not resume after tick {tick:?}")]
    TickResume { tick: TickInfo, source: SpineError },
    #[error("could not resume after tick {tick:?}")]
    TickNotApp { tick: TickInfo, got: &'static str },
    #[error("missing FFI symbol: {0}")]
    BadDyn(String),
    #[error("IO monad did not return value after {0} steps")]
    IOIncomplete(usize),
}

/// INVARIANT: At all times, `watermark <= stack.len()`.
#[derive(Collect, Default)]
#[collect(no_drop)]
struct Spine<'gc> {
    stack: Vec<Pointer<'gc>>,
    watermark: usize,
}

impl<'gc> Spine<'gc> {
    fn check_invariant(&self) {
        debug_assert!(
            self.watermark <= self.stack.len(),
            "watermark ({}) should be less than or equal to spine length ({})",
            self.watermark,
            self.stack.len()
        );
    }

    fn size(&self) -> usize {
        self.check_invariant();
        self.stack.len() - self.watermark
    }

    fn has_args(&self, num: usize) -> bool {
        self.check_invariant();
        // use saturating sub here to avoid bounds check
        self.watermark <= self.stack.len().saturating_sub(num)
    }

    fn push(&mut self, app: Pointer<'gc>) {
        self.stack.push(app);
        self.check_invariant();
    }

    fn pop(&mut self) -> Result<Pointer<'gc>, SpineError> {
        self.check_invariant();
        self.stack.pop().ok_or(SpineError::NoPop)
    }

    fn peek(&self, idx: usize) -> Result<Pointer<'gc>, SpineError> {
        self.check_invariant();
        let idx = self.stack.len() - 1 - idx;
        self.stack.get(idx).cloned().ok_or(SpineError::NoPeek(idx))
    }

    fn save(&mut self) -> usize {
        let prev = self.watermark;
        self.watermark = self.stack.len();
        self.check_invariant();
        prev
    }

    fn restore(&mut self, prev: usize) {
        self.check_invariant();
        self.stack.drain(self.watermark..);
        self.watermark = prev;
        self.check_invariant();
    }
}

/// Pending strict evaluation some combinator's arguments.
#[derive(Collect, Clone, Copy)]
#[collect(require_static)]
struct StrictWork {
    op: Operator,
    args: BitArr!(for Self::MAX_ARGS),
    watermark: usize,
}
impl Debug for StrictWork {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut bitslice: [bool; Self::MAX_ARGS] = [false; Self::MAX_ARGS];
        for (i, b) in bitslice.iter_mut().enumerate() {
            *b = self.args[i];
        }
        f.debug_struct("StrictWork")
            .field("op", &self.op)
            .field("args", &bitslice)
            .field("watermark", &self.watermark)
            .finish()
    }
}

impl StrictWork {
    const MAX_ARGS: usize = 4;

    fn init<'gc>(
        mc: &Mutation<'gc>,
        op: Operator,
        spine: &mut Spine<'gc>,
    ) -> VMResult<Option<(Self, Pointer<'gc>)>> {
        log::trace!(
            "Evaluated tip to {op}, checking whether {} arguments need to be evaluated.",
            op.arity()
        );
        debug_assert!(
            spine.has_args(op.strictness().len()),
            "spine should have at least {} arguments when starting strict work for {op}",
            op.strictness().len(),
        );

        // Figure out which args need to be evaluated, i.e., are not yet WHNF
        // Default state: we don't need to evalute each arg
        let mut args = BitArray::new([0]);

        // Also find index of first argument that needs evaluating
        let first = OnceCell::new();

        for (idx, is_strict) in op.strictness().iter().enumerate() {
            if !*is_strict {
                #[cfg(debug_assertions)]
                {
                    let arg = spine.peek(idx).unwrap();
                    assert!(
                        arg.unpack().is_app(),
                        "argument {idx} of {op} should be an app node, but it was {arg:?}"
                    );
                }
                log::trace!("..... Arg {idx} is not strict, no need to check");
                continue;
            }
            let app = expect!(spine.peek(idx)).forward(mc);
            let arg = unpack_arg(app, idx, op)?.forward(mc);
            if !arg.unpack().whnf() {
                // Remember the first argument that needs evaluating:
                let is_pending = first.set(idx).is_err();
                if is_pending {
                    // If first.set().is_err(), then this argument is pending after `arg[first]`
                    // is evaluated, so we mark it as such:
                    args.set(idx, true);
                }
                log::trace!("..... Arg {idx} is strict and not already in WHNF: {arg:?}");
            } else {
                log::trace!("..... Arg {idx} is strict, but already in WHNF: {arg:?}");
            }
        }

        if let Some(first) = first.get() {
            let next = expect!(spine.peek(*first)); // No need to forward, already done in loop
            let watermark = spine.save();
            let work = Self {
                op,
                args,
                watermark,
            };
            log::trace!("==> Next, evaluating argument {first} of {op}: {next:?}");
            log::trace!("    with pending work {work:?}");
            Ok(Some((work, next)))
        } else {
            log::trace!("... No arguments of {op} need to be evaluated");
            Ok(None)
        }
    }

    /// Advance the pending arg state.
    fn advance<'gc>(
        &mut self,
        mc: &Mutation<'gc>,
        spine: &mut Spine<'gc>,
    ) -> VMResult<Option<Pointer<'gc>>> {
        log::trace!("... Resuming {:?}", self);
        debug_assert!(
            spine.has_args(self.op.strictness().len()),
            "spine should have at least {} arguments when starting strict work for {}",
            self.op.strictness().len(),
            self.op
        );
        debug_assert!(!self.args[0], "index 0 should never be set");
        for idx in 1..Self::MAX_ARGS {
            if self.args[idx] {
                self.args.set(idx, false);
                let app = expect!(spine.peek(idx)).forward(mc);
                let arg = unpack_arg(app, idx, self.op)?.forward(mc);
                if !arg.unpack().whnf() {
                    log::trace!("..... Arg {idx} will be evaluated next: {arg:?}");
                    return Ok(Some(arg));
                }
                log::trace!(
                    "..... Arg {idx} was previously scheduled, but it is now in WHNF: {arg:?}"
                );
            }
        }
        log::trace!(
            "... No more work needs to be done, all {} arguments in WHNF",
            self.op.arity()
        );
        spine.restore(self.watermark);
        Ok(None)
    }
}

#[derive(Clone, Copy, Collect, Debug)]
#[collect(require_static)]
pub enum IO {
    Bind { cc: bool },
    Then,
    Perform,
}
impl From<IO> for Combinator {
    fn from(value: IO) -> Self {
        match value {
            IO::Bind { cc } => {
                if cc {
                    Combinator::CCBind
                } else {
                    Combinator::Bind
                }
            }
            IO::Then => Combinator::Then,
            IO::Perform => Combinator::PerformIO,
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

    fn pop_arg<T: FromValue<'gc> + Debug>(
        &mut self,
        mc: &Mutation<'gc>,
        index: usize,
        op: Operator,
    ) -> VMResult<(Pointer<'gc>, T)> {
        let app = expect!(self.spine.pop()).forward(mc);
        let arg = unpack_arg(app, index, op)?.forward(mc);
        let arg = T::from_value(mc, arg.unpack()).map_err(|source| VMError::TypeError {
            op,
            index,
            source,
        })?;
        log::trace!(
            "..... Popped argument {index} of type {} from spine: {arg:?}",
            core::any::type_name::<T>()
        );
        Ok((app, arg))
    }

    /// Symbolic handler for combinators with well-defined "redux code".
    ///
    /// With luck, the `rustc` optimizer will unroll this stack machine interpreter into efficient
    /// code, such that the `args` are kept in registers, and the `stk` is completely gone.
    ///
    /// Note that this function assumes the following:
    ///
    /// -   `comb` is a combinator with `Some` well-formed [`ReduxCode`]
    /// -   `comb`'s `ReduxCode` does not require more than 8 stack slots
    /// -   `comb` does receive more than 8 arguments (`arity` is less than 8)
    /// -   there are enough arguments in the spine, due to the runtime check of `has_args()`
    ///     in `start_strict()`
    ///
    /// These can all be statically verified, and does not rely on the correctness of how any
    /// combinator is handled. Thus, it should not panic.
    ///
    /// However, this function _may_ throw an error if one of the arguments popped from the spine
    /// is not actually an `App` node. Ideally, we would statically verify that this scenario never
    /// happens either.
    #[inline(always)]
    #[track_caller]
    fn do_redux(&mut self, mc: &Mutation<'gc>, comb: Combinator) -> VMResult<Pointer<'gc>> {
        log::trace!("... Performing reduction for {comb:?}");
        debug_assert!(
            self.spine.has_args(comb.strictness().len()),
            "spine should have at least {} arguments when starting strict work for {comb}",
            comb.strictness().len(),
        );

        let code = expect!(comb.redux(), "only equational combinators can be reduced");
        let mut top = self.tip;
        let mut args = heapless::Vec::<_, 8>::new();
        let mut stk = heapless::Vec::<_, 8>::new();

        for index in 0..comb.arity() {
            let app = expect!(self.spine.pop());
            let arg = unpack_arg(app, index, comb)?;
            expect!(args.push(arg));
            top = app;
            log::trace!("..... Argument {index}: {arg:?}");
        }

        for b in &code[..code.len() - 1] {
            match b {
                ReduxCode::Arg(i) => expect!(stk.push(args[*i])),
                ReduxCode::Top => expect!(stk.push(top)),
                ReduxCode::App => {
                    let arg = expect!(stk.pop());
                    let fun = expect!(stk.pop());
                    expect!(stk.push(Pointer::new(mc, App { fun, arg })))
                }
            }
        }

        // We need to handle the last argument differently
        let next = match code[code.len() - 1] {
            ReduxCode::Arg(i) => top.set_ref(mc, args[i]),
            ReduxCode::Top => unreachable!(
                "{comb} redux code not well-formed: no rule should reduce to '^' alone"
            ),
            ReduxCode::App => {
                let arg = expect!(stk.pop());
                let fun = expect!(stk.pop());
                top.set(mc, App { fun, arg });
                top
            }
        };
        debug_assert!(
            stk.is_empty(),
            "{comb} redux code not well-formed: stack should be empty after reduction"
        );
        log::trace!("==> Reduced to {next:?}");
        Ok(next)
    }

    /// Handle one of the core [`IO`] combinators.
    ///
    /// Like [`State::do_redux()`], this function should not
    fn start_io(&mut self, mc: &Mutation<'gc>, io: IO) -> VMResult<Pointer<'gc>> {
        let comb = Combinator::from(io);
        debug_assert!(
            // not necessary, callers did this already
            self.spine.has_args(comb.arity()),
            "spine should have at least {} arguments when reducing IO node {comb}",
            comb.arity()
        );
        let top = expect!(self.spine.pop());
        let mut next = unpack_arg(top, 0, comb)?;

        // Prepare spine for continuation
        match io {
            IO::Bind { cc } => {
                let app = expect!(self.spine.pop());
                let continuation = unpack_arg(app, 1, comb)?;
                if cc {
                    let app = expect!(self.spine.pop());
                    let k = unpack_arg(app, 2, comb)?;
                    next = Pointer::new(mc, App { fun: next, arg: k });
                }
                self.spine.push(continuation);
                log::trace!("==> {io:?} context saved, with continuation {continuation:?}");
                log::trace!("    Evaluating next: {next:?}");
            }
            IO::Then => {
                let app = expect!(self.spine.pop());
                let continuation = unpack_arg(app, 1, comb)?;
                self.spine.push(continuation);
                log::trace!("==> {io:?} context saved, with continuation {continuation:?}");
                log::trace!("    Evaluating next: {next:?}");
            }
            IO::Perform => {
                // Push the top app node back onto the spine. We will overwrite it later.
                self.spine.push(top);
                log::trace!("==> {io:?} context saved, will unwrap next: {next:?}");
            }
        }
        self.io_context.push(io);
        Ok(next)
    }

    fn resume_io(&mut self, mc: &Mutation<'gc>, value: Pointer<'gc>) -> VMResult<Pointer<'gc>> {
        let Some(io) = self.io_context.pop() else {
            log::trace!("==> No IO context to handle {value:?}");
            return Err(value.follow().unpack().try_unwrap_combinator().map_or_else(
                |v| VMError::IONoContext(v.input.into()), // terminated with non-combinator value
                VMError::IOTerminated,                    // terminated with combinator constant
            ));
        };
        let continuation = self
            .spine
            .pop()
            .map_err(|source| VMError::IONoContinuation {
                io: io.into(),
                source,
            })?;

        let next = match io {
            IO::Bind { .. } => Pointer::new(
                mc,
                App {
                    fun: continuation,
                    arg: value,
                },
            ),
            IO::Then => {
                // Ignore the monadic value, return the continuation
                continuation
            }
            IO::Perform => {
                // Set the top node applying PerformIO to the resulting value
                continuation.set_ref(mc, value)
            }
        };
        log::trace!("==> Resuming in {io:?} context with continuation {continuation:?}");
        Ok(next)
    }

    fn resume_pure(
        &mut self,
        mc: &Mutation<'gc>,
        next: Pointer<'gc>,
        value: impl IntoValue<'gc>,
    ) -> VMResult<Pointer<'gc>> {
        next.set(mc, value);
        log::trace!("==> Resuming in pure context with next {next:?}");
        Ok(next)
    }

    fn handle_unop<P, R>(
        &mut self,
        mc: &Mutation<'gc>,
        op: impl Into<Operator>,
        handler: impl FnOnce(P) -> R,
    ) -> VMResult<Pointer<'gc>>
    where
        P: Debug + Copy + FromValue<'gc>,
        R: IntoValue<'gc>,
    {
        let op: Operator = op.into();
        debug_assert!(op.arity() == 1, "{op} should be a unary operator");

        let (top, arg0) = self.pop_arg::<P>(mc, 0, op)?;
        let value = handler(arg0).into_value(mc);
        log::trace!(
            "... {op} returned value of type {}: {value:?}",
            core::any::type_name::<R>()
        );

        if op.io_action() {
            self.resume_io(mc, Pointer::new(mc, value))
        } else {
            self.resume_pure(mc, top, value)
        }
    }

    fn handle_binop<P, Q, R>(
        &mut self,
        mc: &Mutation<'gc>,
        op: impl Into<Operator>,
        handler: impl FnOnce(P, Q) -> R,
    ) -> VMResult<Pointer<'gc>>
    where
        P: Debug + Copy + FromValue<'gc>,
        Q: Debug + Copy + FromValue<'gc>,
        R: IntoValue<'gc>,
    {
        let op: Operator = op.into();
        debug_assert!(op.arity() == 2, "{op} should be a binary operator");

        let (_, arg0) = self.pop_arg::<P>(mc, 0, op)?;
        let (top, arg1) = self.pop_arg::<Q>(mc, 1, op)?;
        let value = handler(arg0, arg1).into_value(mc);
        log::trace!(
            "... {op} returned value of type {}: {value:?}",
            core::any::type_name::<R>()
        );

        if op.io_action() {
            self.resume_io(mc, Pointer::new(mc, value))
        } else {
            self.resume_pure(mc, top, value)
        }
    }

    fn handle_from_utf8(&mut self, mc: &Mutation<'gc>) -> VMResult<Pointer<'gc>> {
        let comb = Combinator::FromUtf8;
        let (top, s) = self.pop_arg::<Value>(mc, 0, comb.into())?;
        let s = s.try_unwrap_string().map_err(|got| VMError::TypeError {
            op: comb.into(),
            index: 0,
            source: ConversionError {
                expected: "string",
                got: got.input.into(),
            },
        })?;
        let res = crate::string::hstring_from_utf8(mc, s.as_ref());
        log::trace!("... {comb:?} returned {res:?}");
        self.resume_pure(mc, top, res)
    }

    fn handle_new_castring(&mut self, mc: &Mutation<'gc>) -> VMResult<Pointer<'gc>> {
        let comb = Combinator::CAStringLenNew;
        let top = expect!(self.spine.pop()).forward(mc);
        let s = unpack_arg(top, 0, comb)?.forward(mc);

        if let Some(cas) = new_castringlen(mc, s) {
            log::trace!("... {comb:?} returned {cas:?}");
            self.resume_io(mc, cas)
        } else {
            let new = Pointer::new(mc, comb);
            let again = Pointer::new(mc, App { fun: new, arg: s });
            log::trace!("... {comb:?} arguments were not fully-evaluated, scheduling <$> seq before returning to {again:?}");
            self.resume_pure(mc, top, eval_hstring(mc, s, again))
        }
    }

    /// `next` has been reduced to the given combinator, with all strict arguments evaluated.
    fn eval_comb(&mut self, comb: Combinator, mc: &Mutation<'gc>) -> VMResult<Pointer<'gc>> {
        log::trace!("Handling combinator reduction: {comb:?}");
        debug_assert!(
            self.spine.has_args(comb.arity()),
            "spine should have at least {} arguments when evaluating {comb}",
            comb.arity(),
        );

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

            Combinator::FromUtf8 => self.handle_from_utf8(mc),
            Combinator::CAStringLenNew => self.handle_new_castring(mc),
            Combinator::CAStringPeek => self.handle_unop(mc, comb, peek_castring),
            Combinator::CAStringPeekLen => self.handle_binop(mc, comb, peek_castringlen),
        }
    }

    fn eval_ffi(&mut self, ffi: FfiSymbol, mc: &Mutation<'gc>) -> VMResult<Pointer<'gc>> {
        log::trace!("Handling FFI call: {ffi}");
        debug_assert!(
            self.spine.has_args(ffi.arity()),
            "spine should have at least {} arguments when evaluating {ffi}",
            ffi.arity(),
        );

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

    fn start_strict(&mut self, op: Operator, mc: &Mutation<'gc>) -> VMResult<Pointer<'gc>> {
        if !self.spine.has_args(op.arity()) {
            log::trace!(
                "Evaluated tip to {op} (arity = {}), but spine only has {} arguments left",
                op.arity(),
                self.spine.size()
            );
            log::trace!(
                "==> treating partially-applied {op} as a value and resuming strict evaluation."
            );
            return self.resume_strict(mc);
        }

        if let Some((work, first)) = StrictWork::init(mc, op, &mut self.spine)? {
            self.strict_context.push(work);
            Ok(first)
        } else {
            // If start_strict returns None, then we can directly handle the combinator
            match op {
                Operator::Combinator(comb) => self.eval_comb(comb, mc),
                Operator::FfiSymbol(ffi) => self.eval_ffi(ffi, mc),
            }
        }
    }

    fn resume_strict(&mut self, mc: &Mutation<'gc>) -> VMResult<Pointer<'gc>> {
        // `tip` has been reduced to a value, probably because some strict operator had
        // previously demanded strict arguments. We check our strict continuation context
        // to see if we still need to evaluate more arguments, or if we can now handle the
        // strict operator.
        log::trace!("Evaluated tip to value {:?}", self.tip.unpack());
        let work = self
            .strict_context
            .last_mut()
            .ok_or(VMError::UnexpectedValue(self.tip.unpack().into()))?;

        if let Some(next) = work.advance(mc, &mut self.spine)? {
            Ok(next)
        } else {
            let op = work.op;
            self.strict_context.pop();
            match op {
                Operator::Combinator(comb) => self.eval_comb(comb, mc),
                Operator::FfiSymbol(ffi) => self.eval_ffi(ffi, mc),
            }
        }
    }

    fn handle_tick(&mut self, t: Tick) -> Result<Pointer<'gc>, VMError> {
        let tick = self.tick_table.tick(t)?;
        let app = self.spine.pop().map_err(|source| VMError::TickResume {
            tick: tick.clone(),
            source,
        })?;
        if let Value::App(App { fun: _, arg }) = app.unpack() {
            log::trace!("  ! Handled tick: {tick:?}");
            log::trace!("    and resuming execution at {arg:?}");
            Ok(arg)
        } else {
            Err(VMError::TickNotApp {
                tick: self.tick_table.info(t).clone(),
                got: app.unpack().into(),
            })
        }
    }

    fn step(&mut self, mc: &Mutation<'gc>) -> VMResult<()> {
        // Tight loop for non-GC-allocating steps
        loop {
            match self.tip.unpack() {
                Value::App(App { fun, arg: _ }) => {
                    log::trace!("  @ Pushing {:?} to spine", self.tip);
                    self.spine.push(self.tip);
                    self.tip = fun;
                }
                Value::Ref(_) => {
                    log::trace!("  * Forwarding {:?}", self.tip);
                    self.tip.forward(mc);
                }
                Value::Tick(t) => {
                    self.tip = self.handle_tick(t)?;
                }
                _ => break,
            }
        }

        self.tip = match self.tip.unpack() {
            Value::Ref(_) | Value::App(_) | Value::Tick(_) => {
                unreachable!("unexpected: {:?}", self.tip)
            }
            Value::Combinator(comb) => self.start_strict(comb.into(), mc),
            Value::String(_) | Value::Integer(_) | Value::Float(_) | Value::ForeignPtr(_) => {
                self.resume_strict(mc)
            }
            Value::Ffi(ffi) => self.start_strict(ffi.into(), mc),
            Value::BadDyn(sym) => Err(VMError::BadDyn(sym.name().to_string())),
        }?;
        Ok(())
    }

    fn steps(&mut self, mc: &Mutation<'gc>, mut bound: usize) -> VMResult<()> {
        while bound > 0 {
            self.step(mc)?;
            bound -= 1;
        }
        Ok(())
    }

    fn complete(&mut self, mc: &Mutation<'gc>, bound: usize) -> VMResult<()> {
        match self.steps(mc, bound) {
            Err(VMError::IOTerminated(Combinator::I)) => Ok(()),
            Ok(()) => Err(VMError::IOIncomplete(bound)),
            Err(e) => Err(e),
        }
    }
}

pub struct VM {
    arena: Arena<Rootable![State<'_>]>,
}

impl VM {
    pub fn step(&mut self) -> VMResult<()> {
        self.arena.mutate_root(|mc, st| st.step(mc))
    }
    pub fn steps(&mut self, bound: usize) -> VMResult<()> {
        self.arena.mutate_root(|mc, st| st.steps(mc, bound))
    }
    pub fn complete(&mut self, bound: usize) -> VMResult<()> {
        self.arena.mutate_root(|mc, st| st.complete(mc, bound))
    }
}

impl Debug for VM {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
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
            Pointer::new($mc, Combinator::$c)
        };
    }
    macro_rules! int {
        ($mc:ident, $i:literal) => {
            Pointer::new($mc, Integer::from($i))
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
            Pointer::new($mc, App { fun: $f, arg: $a })
        };
        ($mc:ident, $f:expr, $a:expr, $($rest:tt)+) => {
            app!($mc, Pointer::new($mc, App { fun: $f, arg: $a }), $($rest)+)
        };
    }

    #[test]
    fn arith420() {
        type Root = Rootable![(State<'_>, Pointer<'_>)];
        let mut arena: Arena<Root> = Arena::new(|mc| {
            let top = app!(mc, :Mul, #42, #10);
            (State::new(top, Default::default()), top)
        });

        arena.mutate_root(|mc, (st, _)| st.steps(mc, 1)).unwrap();
        // 1 reducing step should be sufficient:
        //      @ tip points to ((* 42) 10)
        //      @ tip points to (* 42)
        //      @ tip points to *
        //      ! reduce *, tip points to result (420)
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
        type Root = Rootable![(State<'_>, Pointer<'_>)];
        let mut arena: Arena<Root> = Arena::new(|mc| {
            let top = app!(mc, :Mul, @(:Add, #40, #2), #10);
            (State::new(top, Default::default()), top)
        }); // tip points to ((* ((+ 40) 2)) 10)
        arena.mutate_root(|mc, (st, _)| st.step(mc)); // tip points to (* ((+ 40) 2))
        arena.mutate_root(|mc, (st, _)| st.step(mc)); // tip points to *
        arena.mutate_root(|mc, (st, _)| st.step(mc)); // start strict *, tip points to ((+ 40) 2)
        arena.mutate_root(|mc, (st, _)| st.step(mc)); // tip points to (+ 40)
        arena.mutate_root(|mc, (st, _)| st.step(mc)); // tip points to +
        arena.mutate_root(|mc, (st, _)| st.step(mc)); // reduce +, tip points to 42
        arena.mutate_root(|mc, (st, _)| st.step(mc)); // resume strict *, tip points to result 420
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
    fn skkr() {
        type Root = Rootable![(State<'_>, Pointer<'_>)];
        let mut arena: Arena<Root> = Arena::new(|mc| {
            let (s, k, r) = (comb!(mc, S), comb!(mc, K), comb!(mc, R));
            let top = app!(mc, s, k, k, r);
            (State::new(top, Default::default()), top)
        });

        arena.mutate_root(|mc, (st, _)| st.steps(mc, 2)).unwrap();
        // 2 reducing steps should be sufficient:
        //
        //      @ tip points to (((S K) K) R)
        //      @ tip points to ((S K) K)
        //      @ tip points to (S K)
        //      @ tip points to S
        //      ! reduce S, tip points to ((K R) (K R))
        //      @ tip points to (K R)
        //      @ tip points to K
        //      ! reduce K, tip points to *->R
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

        arena.mutate_root(|mc, (st, _)| st.steps(mc, 3)).unwrap();
        //      @ tip points to ((>>= ((>> (return_0 bottom)) (return_1 result))) return_2)
        //      @ tip points to (>>= ((>> (return_0 bottom)) (return_1 result)))
        //      @ tip points to >>=
        //      ! reduce >>=, tip points to ((>> (return_0 bottom)) (return_1 result))
        //      @ tip points to (>> (return_0 bottom))
        //      @ tip points to >>
        //      ! reduces >>, tip points to (return_1 result)
        //      @ tip points to return_1
        //      ! reduce return_1, tip points to (return_2 result)

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
