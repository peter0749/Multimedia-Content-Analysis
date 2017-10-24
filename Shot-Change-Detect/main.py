import sys
import argparse
import detector
import benchmark

parser = argparse.ArgumentParser(description='Simple Shot Change Detection using Python+OpenCV')
parser.add_argument('path', metavar='file_path', type=str,
                    help='Path to the raw frames')
parser.add_argument('--threshold', type=float, default=0.8, required=False,
                    help='Threshold of shot change')
parser.add_argument('--min_length', type=int, default=12, required=False,
                    help='Minimum frame length of one scene.')
parser.add_argument('--read_type', type=str, default='video', required=False,
        help='Read from raw frames (from directory) or a video file. (default: video)')
parser.add_argument('--bench', type=float, default=-1, required=False,
        help='Measure performance')
parser.add_argument('--bench_upper_bound', type=float, default=1, required=False,
        help='Measure performance')
parser.add_argument('--ground_truth', type=str, default='', required=False,
        help='Path to ground truth')
parser.add_argument('--keyframe_out', type=str, default='', required=False,
        help='Path to generated keyframes')
parser.add_argument('--keyframe_threshold', type=float, default=0.82, required=False,
        help='Keyframe threshold')
parser.add_argument('--scale', type=float, default=None, required=False,
        help='Subsampling')

if __name__ == '__main__':
    args = parser.parse_args()
    SCD = detector.ContentBased(directory = args.path, img_type=args.read_type, scale=args.scale)
    if args.bench<=0:
        sc = SCD.run(threshold = args.threshold, min_length = args.min_length)
        for i in sc: print(i)
        if len(args.keyframe_out)>0:
            SCD.get_keyframe(args.keyframe_threshold, args.keyframe_out)
    else:
        ground_truth = benchmark.gt_parser(args.ground_truth)
        benchmarker = benchmark.benchmark(SCD, ground_truth , args.bench, args.bench_upper_bound, args.min_length)
        benchmarker.run()
