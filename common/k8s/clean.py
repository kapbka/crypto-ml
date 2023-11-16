from kubernetes import client, config
from kubernetes.stream import stream


def main():
    # run locally uncomment the next call
    # config.load_kube_config('crypto-ml/k8s/cluster/credentials/ml.conf')
    config.load_incluster_config()
    v1 = client.CoreV1Api()
    result = v1.list_pod_for_all_namespaces(watch=False, label_selector='k8s-app=registry')
    name = result.items[0].metadata.name

    resp = stream(v1.connect_get_namespaced_pod_exec,
                  name,
                  'kube-system',
                  command=['/bin/registry', 'garbage-collect', '/etc/docker/registry/config.yml'],
                  stderr=True, stdin=False,
                  stdout=True, tty=False)

    print(resp)


if __name__ == '__main__':
    main()
