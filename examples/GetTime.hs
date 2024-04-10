module GetTime(main) where
import Prelude
import Data.IOArray
import Data.IORef
import System.IO
default (String)

main :: IO ()
main = do
  t <- getTimeMilli
  putStrLn $ "The time is: " ++ show t
