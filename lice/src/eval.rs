use gc_arena::{Collect, Gc, Mutation, Rootable};

use crate::{
    combinator::Combinator,
    memory::{Node, Pointer, Value},
};

/// Continuation state, stored in the stack
#[derive(Collect)]
#[collect(no_drop)]
enum Continuation<'gc> {
    Arg(Pointer<'gc>),
}

#[derive(Collect)]
#[collect(no_drop)]
struct State<'gc> {
    stack: Vec<Continuation<'gc>>,
    next: Pointer<'gc>,
}

type Root = Rootable![State<'_>];

impl<'gc> State<'gc> {
    #[inline]
    fn pop_arg(&mut self) -> (Pointer<'gc>, Pointer<'gc>) {
        let Some(Continuation::Arg(app)) = self.stack.pop() else {
            panic!("insufficient arguments")
        };
        let Value::App { fun: _, arg } = app.unpack() else {
            unreachable!()
        };
        (app, arg)
    }

    fn eval_one(&mut self, mc: &Mutation<'gc>) {
        let cur = &self.next;
        self.next = match cur.unpack() {
            Value::App { fun, arg: _arg } => {
                self.stack.push(Continuation::Arg(*cur));
                fun
            }
            Value::Ref(ptr) => {
                // Forward indirection here?
                ptr
            }
            Value::String(s) => todo!("what to do with string {s}?"),
            Value::Integer(i) => todo!("what to do with integer {i}?"),
            Value::Float(f) => todo!("what to do with float {f}?"),
            Value::Combinator(c) => match c {
                Combinator::S => {
                    let (_, f) = self.pop_arg();
                    let (_, g) = self.pop_arg();
                    let (top, x) = self.pop_arg();
                    let fx = Pointer::new(mc, Value::App { fun: f, arg: x });
                    let gx = Pointer::new(mc, Value::App { fun: g, arg: x });
                    top.set(mc, Value::App { fun: fx, arg: gx });
                    top
                }
                Combinator::K => {
                    let (_, x) = self.pop_arg();
                    let (top, _y) = self.pop_arg();
                    top.set(mc, Value::Ref(x));
                    top
                }
                Combinator::I => {
                    let (top, x) = self.pop_arg();
                    top.set(mc, Value::Ref(x));
                    top
                }
                Combinator::B => {
                    let (_, f) = self.pop_arg();
                    let (_, g) = self.pop_arg();
                    let (top, x) = self.pop_arg();
                    let gx = Pointer::new(mc, Value::App { fun: g, arg: x });
                    top.set(mc, Value::App { fun: f, arg: gx });
                    top
                }
                Combinator::C => {
                    let (_, f) = self.pop_arg();
                    let (_, x) = self.pop_arg();
                    let (top, y) = self.pop_arg();
                    let fy = Pointer::new(mc, Value::App { fun: f, arg: y });
                    top.set(mc, Value::App { fun: fy, arg: x });
                    top
                }
                _ => unimplemented!("handler for combinator {c}"),
            },
        }
    }
}
