from iemocap import *
from multitask_training_session import *
from models_multi_task import *

if __name__ == '__main__':
    iemocap_four_noprep_test = IemocapDataset(  # Без препроцессинга, тестовый
        PATH_TO_PICKLE, IEMOCAP_PATH_TO_WAVS, IEMOCAP_PATH_TO_EGEMAPS, IEMOCAP_PATH_FOR_PARSER,
        base_name='IEMOCAP-4', label_type='four', mode='test', preprocessing=False,
        augmentation=False, padding='repeat', spectrogram_shape=224, spectrogram_type='melspec', tasks=['emotion']
    )
    model = AlexNetMultiTask(4, 10, 2)
    dataset = iemocap_four_noprep_test
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters())
    device = torch.device('cpu')
    ts = TrainingSession(name='FirstTry',
                         model=model,
                         dataset=dataset,
                         criterion=criterion,
                         optimizer=optimizer,
                         num_epochs=100,
                         batch_size=32,
                         device=device,
                         path_to_weights=WEIGHTS_FOLDER,
                         path_to_results=RESULTS_FOLDER)

    ts.overfit_one_batch(100, 10)

