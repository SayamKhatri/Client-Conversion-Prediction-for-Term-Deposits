from kfp.v2.dsl import pipeline
from kubeflow_pipeline.components.get_data.get_data import get_data
from kubeflow_pipeline.components.prepare_data_for_rf.prepare_data_for_rf import prepare_data_for_rf
from kubeflow_pipeline.components.prepare_data_for_nn.prepare_data_for_nn import prepare_data_for_nn
from kubeflow_pipeline.components.train_rf_model.train_rf_model import train_rf_model
from kubeflow_pipeline.components.train_nn_model.train_nn_model import train_nn_model
from kubeflow_pipeline.components.evaluate_rf.evaluate_rf import evaluate_rf
from kubeflow_pipeline.components.evaluate_nn.evaluate_nn import evaluate_nn
from kubeflow_pipeline.components.select_best_model.select_best_model import select_best_model
from kubeflow_pipeline.components.register_model.register_model import register_model
from kubeflow_pipeline.components.deploy_model.deploy_model import deploy_model
from kubeflow_pipeline.config import PROJECT_ID, REGION, BUCKET_NAME

@pipeline(
    name = 'bank-policy-production-f',
    pipeline_root = f'gs://{BUCKET_NAME}/pipeline-artifacts'
)
def pipeline():
    get_data_task = get_data(raw_gcs_url=f'gs://{BUCKET_NAME}/data/bank-full.csv')
    prepare_rf_task = prepare_data_for_rf(raw_dataset=get_data_task.outputs['raw_dataset'])
    prepare_nn_task = prepare_data_for_nn(raw_dataset=get_data_task.outputs['raw_dataset'])
    train_rf_model_task = train_rf_model(train_rf=prepare_rf_task.outputs['train_rf'])
    train_nn_model_task = train_nn_model(train_nn=prepare_nn_task.outputs['train_nn'])
    eval_rf_task = evaluate_rf(
        test_rf=prepare_rf_task.outputs['test_rf'],
        model_rf=train_rf_model_task.outputs['model_rf'],
        project_id=PROJECT_ID,
        region=REGION,
        experiment_name="exprandomforest10prod1",
        run_name="randomeforestrun10prod1"
    )
    eval_nn_task = evaluate_nn(
        test_nn=prepare_nn_task.outputs['test_nn'],
        model_nn=train_nn_model_task.outputs['model_nn'],
        project_id=PROJECT_ID,
        region=REGION,
        experiment_name="expneuralnetwork10prod1",
        run_name="neuralnetworrun10prod1"
    )
    best_model_selection_task = select_best_model(
        metrics_rf=eval_rf_task.outputs['metrics_rf'],
        metrics_nn=eval_nn_task.outputs['metrics_nn'],
        model_rf=train_rf_model_task.outputs['model_rf'],
        model_nn=train_nn_model_task.outputs['model_nn']
    )
    register_model_task = register_model(
        should_deploy=best_model_selection_task.outputs['should_deploy'],
        best_model_path=best_model_selection_task.outputs['best_model_path'],
        best_model_framework=best_model_selection_task.outputs['best_model_framework'],
        project_id=PROJECT_ID,
        region=REGION                                       
    )
    deploy_model_task = deploy_model(
        registered_model=register_model_task.outputs['registered_model'],
        project_id=PROJECT_ID,
        region=REGION,
        endpoint_display_name="bank-model-endpoint",
        deployed_model_display_name="bank-model-deployed",
        machine_type="n1-standard-4"
    )

if __name__ == '__main__':
    from kfp.v2 import compiler
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path='bank_pipeline.json'
    )

    from google.cloud import aiplatform
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_NAME)
    job = aiplatform.PipelineJob(
        display_name='bank-policy-prod',
        template_path='bank_pipeline.json',
        pipeline_root=f'gs://{BUCKET_NAME}/pipeline-artifacts'
    )
    job.run()
