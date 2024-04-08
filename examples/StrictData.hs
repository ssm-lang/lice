module StrictData(main) where
import Prelude
import Primitives
default (Int, Double)

data Foo = Foo !Int Int !Double

-- Construct but throw away a data type with a partially strict constructor
main :: IO ()
main = primSeq (Foo 6 1 9) (primReturn ())
