import json
import sys

if __name__ == '__main__':
    results = list()
    for line in sys.stdin:
        data = json.loads(line)
        del data["_id"]
        data["ts"] = data["ts"]["$date"][:-1]
        results.append(data)

    for r in sorted(results, key=lambda x: x['ts']):
        print(json.dumps(r))
