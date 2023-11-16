import asyncio
import os
import sys

import aiohttp

TOKEN = os.getenv('TOKEN') or open('/var/run/secrets/kubernetes.io/serviceaccount/token', 'r').read()
NAMESPACE = os.getenv('NAMESPACE') or open('/var/run/secrets/kubernetes.io/serviceaccount/namespace', 'r').read()


def get_url(namespace: str):
    prefix = f"https://{os.environ['KUBERNETES_SERVICE_HOST']}:{os.environ['KUBERNETES_PORT_443_TCP_PORT']}"
    return f"{prefix}/api/v1/namespaces/{namespace}/pods"


def is_ready(pod: dict) -> bool:
    return len(list(filter(lambda x: x['type'] == 'Ready' and x['status'] == 'True',
                           pod.get('status', {}).get('conditions', [])))) > 0


async def main(deployment_name: str):
    async with aiohttp.ClientSession() as session:
        url = get_url(NAMESPACE)
        last_pending = []
        while True:
            async with session.get(url,
                                   headers={"Authorization": f"Bearer {TOKEN}"},
                                   ssl=False,
                                   params={"labelSelector": f"app={deployment_name}"}) as response:
                result = await response.json()
                if not isinstance(result.get('items'), list):
                    print(result)
                else:
                    ready = list(map(is_ready, result['items']))
                    if all(ready):
                        print(f"Deployment '{deployment_name}' is ready")
                        return
                    else:
                        pending = list(map(lambda x: x[0]['metadata']['name'],
                                           filter(lambda x: not x[1], zip(result['items'], ready))))
                        if not last_pending or pending != last_pending:
                            print(f"Pending pods: {pending}")
                            last_pending = pending

                await asyncio.sleep(1)


if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(asyncio.gather(*list(map(main, sys.argv[1].split(",")))))
