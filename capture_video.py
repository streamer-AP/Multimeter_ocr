import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("filename")
parser.add_argument("--save","-s",action="store_true",default=False)

args = parser.parse_args()

filename = args.filename
write = args.save
print("write is ", write)


def main():

  cap = cv2.VideoCapture(0)
  width = 1280
  height = 720
  fps = 5

  cap.set(3, width)
  cap.set(4, height)

  

  if write:
    outvideo = cv2.VideoWriter()
    sz = (width, height)
    
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    outvideo.open(filename + ".mp4", fourcc, fps, sz, True)
    #outvideo.open('./output.mp4', fourcc, fps, sz, True)

  while(True):
      ret, frame = cap.read()
      if ret:
        cv2.imshow("cam", frame)

        if(write):
          outvideo.write(frame)
            
        if cv2.waitKey(1) & 0xFF==ord("q"):
            break

if __name__ == "__main__":
    main()
