"""python -m kiln_webgpu install-gpu [version] | info"""
import sys

from . import KilnError, install_gpu, runtime


def main(argv):
    try:
        if argv[:1] == ["install-gpu"]:
            print("Installed Dawn's WebGPU package in %s" % install_gpu(argv[1] if len(argv) > 1 else None))
        elif argv[:1] == ["info"]:
            r = runtime()
            print("Node %s on %s %s; %d tasks" % (r["node"], r["platform"], r["arch"], len(r["tasks"])))
        else:
            print("usage: python -m kiln_webgpu install-gpu [version] | info")
            return 2
    except KilnError as err:
        print("kiln: %s" % err, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
