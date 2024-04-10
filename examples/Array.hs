-- | A simple example that uses IO Arrays.
--
-- It should print:
--
--  10
--  [0,1,4,9,16,25,36,49,64,81]
--  "foo"
--  "bar"
module Array(main) where
import Prelude
import Data.IOArray
import Data.IORef
import System.IO
import System.IO.Serialize
default (String)

main :: IO ()
main = do
  a <- newIOArray 10 0
  s <- sizeIOArray a
  print s
  mapM_ (\ i -> writeIOArray a i (i*i)) [0..9]
  xs <- mapM (readIOArray a) [0..9]
  print xs

  r <- newIORef "foo"
  s1 <- readIORef r
  print s1
  writeIORef r "bar"
  s2 <- readIORef r
  print s2
