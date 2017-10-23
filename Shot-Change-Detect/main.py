import sys
import argparse
import detector

parser = argparse.ArgumentParser(description='Simple Shot Change Detection using Python+OpenCV')
parser.add_argument('path', metavar='file_path', type=str,
                    help='Path to the raw frames')
parser.add_argument('--threshold', type=float, default=0.7, required=False,
                    help='Threshold of shot change')
parser.add_argument('--min_length', type=int, default=12, required=False,
                    help='Minimum frame length of one scene.')
parser.add_argument('--read_type', type=str, default='video', required=False,
        help='Read from raw frames (from directory) or a video file. (default: video)')

if __name__ == '__main__':
    args = parser.parse_args()
    SCD = detector.ContentBased(directory = args.path, threshold = args.threshold, min_length = args.min_length, img_type=args.read_type)
    sc = SCD.run()
    for i in sc: print(i)

