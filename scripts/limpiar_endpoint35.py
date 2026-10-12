import boto3,json
from pathlib import Path
r=json.loads(Path('resultados/validacion_entrada.json').read_text())
e=r['endpoint']
s=boto3.client('sagemaker')
c=s.describe_endpoint(EndpointName=e)['EndpointConfigName']
s.delete_endpoint(EndpointName=e)
s.get_waiter('endpoint_deleted').wait(EndpointName=e)
s.delete_endpoint_config(EndpointConfigName=c)
resultado={'endpoint':e,'endpoint_eliminado':True,'configuracion_eliminada':True}
Path('resultados/limpieza_lab35.json').write_text(json.dumps(resultado,indent=2))
print(resultado)
