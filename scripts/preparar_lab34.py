"""Prepara el notebook OFICIAL de la sesión activa; no ejecuta entrenamiento.

Uso en la terminal de SageMaker:
    python preparar_lab34.py en_us/3_4-machinelearning.ipynb
El notebook resultante debe ejecutarse con conda_python3 en AWS Academy.
"""
import argparse
import json
import re
from pathlib import Path


def markdown(texto):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': texto}


def preparar(origen: Path, destino: Path) -> None:
    notebook = json.loads(origen.read_text(encoding='utf-8'))
    if notebook.get('nbformat') != 4:
        raise ValueError('Se requiere un notebook de formato 4.')
    celdas = notebook['cells']
    celdas.insert(0, markdown(
        '# Laboratorio 3.4 — Entrenamiento de un modelo\n\n'
        '**Estudiante: Santiago Perez Garzon**\n\n'
        '**Ejecución:** consultar la verificación al final del notebook. Antes de ejecutar, '
        'verificar que el bucket y el rol corresponden al laboratorio activo.\n\n'
        'Se sigue el flujo oficial de AWS Academy en Amazon SageMaker: '
        'preparación, división de datos, almacenamiento en S3 y entrenamiento '
        'con XGBoost.\n\n'
        '**Uso ético de IA:** la IA asistió en la revisión del código, la '
        'preparación del notebook y la documentación. Los resultados deben '
        'provenir de la ejecución real en AWS y ser revisados por el estudiante.'
    ))
    for celda in celdas:
        if celda['cell_type'] == 'code':
            fuente = ''.join(celda['source'])
            celda['source'] = fuente.replace(
                'http://archive.ics.uci.edu/', 'https://archive.ics.uci.edu/'
            ).replace(
                'requests.get(f_zip, stream=True)',
                'requests.get(f_zip, stream=True, timeout=60)\nr.raise_for_status()'
            )
            celda['outputs'] = []
            celda['execution_count'] = None

    # La instancia del laboratorio incluye SDK 2; se conserva el algoritmo,
    # la instancia, los canales y los hiperparámetros del notebook oficial.
    for celda in celdas:
        if celda['cell_type'] != 'code':
            continue
        fuente = celda['source']
        if 'sagemaker.core.image_uris' in fuente:
            celda['source'] = fuente.replace('sagemaker.core.image_uris', 'sagemaker.image_uris')
        elif 'from sagemaker.train import ModelTrainer' in fuente:
            instancia = re.search(r"instance_type='([^']+)'", fuente).group(1)
            celda['source'] = (
                'import sagemaker\n'
                'from sagemaker import get_execution_role\n'
                'from sagemaker.inputs import TrainingInput\n'
                "s3_output_location=f's3://{bucket}/{prefix}/output/'\n"
                'xgb_model=sagemaker.estimator.Estimator(\n'
                '    image_uri=container, role=get_execution_role(),\n'
                f"    instance_type='{instancia}', instance_count=1,\n"
                '    output_path=s3_output_location, hyperparameters=hyperparams,\n'
                '    sagemaker_session=sagemaker.Session())'
            )
        elif 'train_channel = InputData' in fuente:
            celda['source'] = (
                "train_channel=TrainingInput(f's3://{bucket}/{prefix}/train/', content_type='text/csv')\n"
                "validate_channel=TrainingInput(f's3://{bucket}/{prefix}/validate/', content_type='text/csv')\n"
                "data_channels={'train':train_channel,'validation':validate_channel}"
            )
        elif 'xgb_model.train' in fuente:
            celda['source'] = 'xgb_model.fit(inputs=data_channels, logs=False)'

    explicaciones = [
        ('from sklearn.model_selection', '### División reproducible\n'
         'La semilla 42 permite repetir la división. La estratificación '
         'conserva la proporción de clases: 80 % para entrenamiento, '
         '10 % para validación y 10 % para prueba.'),
        ('bucket=', '### Datos en S3\n'
         'Se utiliza el bucket del notebook oficial de la sesión activa. '
         'Los CSV no incluyen encabezado ni índice y sitúan la etiqueta '
         'en la primera columna. Cada partición usa un prefijo separado.'),
        ('hyperparams=', '### Configuración de XGBoost\n'
         'Se conservan 42 rondas, binary:logistic y AUC. La probabilidad '
         'corresponde a la clase 1 (Abnormal), según el mapeo del notebook. '
         'El AUC evalúa la ordenación de casos positivos y negativos.'),
        ('xgb_model.fit', '### Entrenamiento administrado\n'
         'SageMaker utiliza los canales de entrenamiento y validación y '
         'guarda el artefacto del modelo en S3. El conjunto de prueba '
         'queda reservado para la evaluación posterior.'),
    ]
    for fragmento, texto in explicaciones:
        posicion = next((i for i, c in enumerate(celdas)
                         if c['cell_type'] == 'code' and fragmento in c['source']), None)
        if posicion is None:
            raise ValueError(f'No se encontró la celda esperada: {fragmento}')
        celdas.insert(posicion, markdown(texto))

    celdas.append(markdown(
        '## Verificación de resultados\n'
        'La siguiente celda comprueba que el trabajo de entrenamiento esté '
        'Completed y conservar sus métricas reales. Las métricas de '
        'validación no sustituyen la evaluación sobre prueba.'
    ))
    celdas.append({
        'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
        'source': """import json
from pathlib import Path
job=xgb_model.latest_training_job.describe()
assert job['TrainingJobStatus']=='Completed'
resultado={k:job[k] for k in ('TrainingJobName','TrainingJobStatus','FinalMetricDataList','HyperParameters','ResourceConfig','ModelArtifacts')}
resultado['filas']={'train':len(train),'validate':len(validate),'test':len(test)}
resultado['compatibilidad']='SDK 2 instalado en la instancia; Estimator.fit equivale al flujo ModelTrainer.train de SDK 3'
Path('resultado_lab34.json').write_text(json.dumps(resultado,indent=2,default=str))
print(json.dumps(resultado,indent=2,default=str))""",
    })
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Preparado: {destino}; {len(celdas)} celdas. Sin ejecutar.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('origen', type=Path)
    parser.add_argument('--destino', type=Path)
    argumentos = parser.parse_args()
    salida = argumentos.destino or argumentos.origen.with_name(
        '3_4_entrenamiento_santiago.ipynb'
    )
    if salida.resolve() == argumentos.origen.resolve():
        parser.error('El destino debe ser diferente del notebook oficial.')
    preparar(argumentos.origen, salida)
