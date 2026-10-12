import json
from pathlib import Path
n=json.load(open('en_us/3_5-machinelearning.ipynb'))
def code(s): return {'cell_type':'code','metadata':{},'source':s,'outputs':[],'execution_count':None}
cells=[{'cell_type':'markdown','metadata':{},'source':'# Laboratorio 3.5 — Despliegue y evaluación\n**Santiago Perez Garzon**\n\nUso ético de IA: asistencia en revisión y documentación; salidas reales de AWS.'}]
for c in n['cells']:
 s=''.join(c['source']); c['source']=s
 if c['cell_type']=='code':
  c['outputs']=[];c['execution_count']=None
  c['source']=s.replace('http://archive.ics.uci.edu/','https://archive.ics.uci.edu/').replace('requests.get(f_zip, stream=True)','requests.get(f_zip, stream=True, timeout=60)\nr.raise_for_status()')
  if 'delete_endpoint' in s: c={'cell_type':'markdown','metadata':{},'source':'El endpoint se elimina después de guardar las evidencias.'}
 cells.append(c)
 if c['cell_type']=='code' and 'xgb_model.deploy' in s:
  cells.append(code('''import json
from pathlib import Path
entrada = "63.0,12.5,45.0,50.5,125.0,3.0"
respuesta = xgb_predictor.predict([float(v) for v in entrada.split(',')])
p = float(respuesta.decode().strip() if isinstance(respuesta, bytes) else respuesta)
assert 0 <= p <= 1
validacion = {'entrada':entrada,'probabilidad_abnormal':p,'clase_umbral_05':'Abnormal' if p>=0.5 else 'Normal','clase_umbral_065':'Abnormal' if p>0.65 else 'Normal','endpoint':xgb_predictor.endpoint_name,'estado_endpoint':boto3.client('sagemaker').describe_endpoint(EndpointName=xgb_predictor.endpoint_name)['EndpointStatus']}
Path('validacion_entrada.json').write_text(json.dumps(validacion,indent=2))
print(json.dumps(validacion,indent=2))'''))
cells.append(code('''import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix
y=test['class'].to_numpy(dtype=int)
p=target_predicted['class'].to_numpy(dtype=float)
assert len(y)==len(p)==31
metricas={}
for umbral in (0.5,0.65):
 pred=(p>umbral).astype(int)
 metricas[str(umbral)]={'accuracy':accuracy_score(y,pred),'precision':precision_score(y,pred,zero_division=0),'recall':recall_score(y,pred,zero_division=0),'f1':f1_score(y,pred,zero_division=0),'roc_auc':roc_auc_score(y,p),'matriz_confusion':confusion_matrix(y,pred,labels=[0,1]).tolist()}
job=xgb_model.latest_training_job.describe()
resultado={'validacion_entrada':validacion,'metricas_prueba':metricas,'filas':{'train':len(train),'validate':len(validate),'test':len(test)},'entrenamiento':{'nombre':job['TrainingJobName'],'estado':job['TrainingJobStatus'],'metricas':job['FinalMetricDataList'],'hiperparametros':job['HyperParameters'],'instancia':job['ResourceConfig']['InstanceType']},'transformacion':{'nombre':xgb_transformer.latest_transform_job.name,'estado':boto3.client('sagemaker').describe_transform_job(TransformJobName=xgb_transformer.latest_transform_job.name)['TransformJobStatus']}}
assert resultado['entrenamiento']['estado']=='Completed'
assert resultado['transformacion']['estado']=='Completed'
Path('resultado_lab35.json').write_text(json.dumps(resultado,indent=2,default=str))
print(json.dumps(resultado,indent=2,default=str))'''))
n['cells']=cells
Path('en_us/3_5_despliegue_santiago.ipynb').write_text(json.dumps(n,ensure_ascii=False,indent=2))
print('PREPARADO')
