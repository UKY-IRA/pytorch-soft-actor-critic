import json
import argparse
from verify import display_results

def main():
    '''
    Quick script to reload already generated results saved on a JSON, 
    should probably always save to the JSON and generate visuals later.
    '''
    parser = argparse.ArgumentParser(description="Reload prexisting results saved as JSON files from verification")
    parser.add_argument("json_file", required=True, help="JSON results file from running verification.")
    parser.add_argument("-o", "--output", required=False, default=None, help="Output directory for generated images, defaults to displaying them.")
    args = parser.parse_args()

    results = []
    with open(args.json_file) as json_file:
        results = json.load(json_file)

    for episode, res in enumerate(results):
        display = args.output is None
        display_results(res, episode, display=display, save_path=args.output)

if __name__ == "__main__":
    main()
