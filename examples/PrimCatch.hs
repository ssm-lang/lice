-- | A very basic test case for IO.catch, using only raw primitives.
module PrimCatch(main) where
import Prelude
import Primitives

main :: IO ()
main = do
  primCatch (primSeq (primError "NO") (primReturn ())) (\_ -> primReturn ()) `primBind` primReturn
