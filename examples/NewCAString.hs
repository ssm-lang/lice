module NewCAString(main) where
import Prelude
import Foreign.C.String (newCAStringLen, peekCAStringLen)
import Foreign.Marshal.Alloc (free)

swedify :: Char -> Char
swedify 'y' = 'j'
swedify x = x

main :: IO ()
main = do
  putStrLn "henlo"
  cs <- newCAStringLen $ map swedify "hey!"
  hs <- peekCAStringLen cs
  putStrLn hs
  free $ fst cs
  putStrLn "bye"
