
clear;
close all;

gpudev = gpuDevice(1);
reset(gpudev);
pathdef;


networks_num = 7;
fold_num = 1;
train_str_num = 4;
dataset_num = 4;

for dataset_i = 3%:dataset_num

    switch dataset_i
        case 1
            %% prepare the CovidDataset data %%%%%%%%%%%%%%%%%%%%%%%%%%%%
            [dataTrain, dataTest, labels] = merchData('/home/Rehrearsal_TransferLearning/Data_Bank/CovidDataset/');
            Dataset_name = 'CovidDataset'
        case 2
            %% prepare the SARS-Cov-2(2482) data %%%%%%%%%%%%%%%%%%%%%%%%%%%%
            [dataTrain, dataTest, labels] = merchData('/home/Rehrearsal_TransferLearning/Data_Bank/SARS-CoV-2(2482)/');
            Dataset_name = 'SARS-CoV-2'
        case 3
            %% prepare the TCIA(971) data %%%%%%%%%%%%%%%%%%%%%%%%%%%%
            [dataTrain, dataTest, labels] = merchData('/home/Rehrearsal_TransferLearning/Data_Bank/TCIA(971)/');
            Dataset_name = 'TCIA'
        case 4
            %% prepare the UCSD-AI4H(539) data %%%%%%%%%%%%%%%%%%%%%%%%%%%%
            [dataTrain, dataTest, labels] = merchData('/home/Rehrearsal_TransferLearning/Data_Bank/UCSD-Al4H(539)/');
            Dataset_name = 'UCSD-Al4H'
        otherwise
            disp('Other Dataset!')
    end

    numClasses = length(labels); 

    %% Set the hyperparameters and Define the structure of CNN

    Accuracy_table = zeros(networks_num, fold_num, train_str_num);
    info_all = cell(networks_num, fold_num, train_str_num);
    
    for networks_i = 2:networks_num

        %% Image Resize First Step
        if networks_i < 6
            dataTrain_resized.Data = zeros(224, 224, 3, length(dataTrain.Files));
            for i_train = 1:length(dataTrain.Files)
                A = imread(dataTrain.Files{i_train});
                data_temp = repmat(A,[1 1 3]);
                dataTrain_resized.Data(:, :, :, i_train) =  imresize3(data_temp, [224 224 3]);;
                dataTrain_resized.Labels(i_train) = dataTrain.Labels(i_train);
            end

            dataTest_resized.Data = zeros(224, 224, 3, length(dataTest.Files));
            for i_test = 1:length(dataTest.Files)
                A = imread(dataTest.Files{i_test});
                data_temp = repmat(A,[1 1 3]);
                dataTest_resized.Data(:, :, :, i_test) =  imresize3(data_temp, [224 224 3]);;
                dataTest_resized.Labels(i_test) = dataTest.Labels(i_test);
            end
        else
            dataTrain_resized.Data = zeros(299, 299, 3, length(dataTrain.Files));
            for i_train = 1:length(dataTrain.Files)
                A = imread(dataTrain.Files{i_train});
                data_temp = repmat(A,[1 1 3]);
                dataTrain_resized.Data(:, :, :, i_train) =  imresize3(data_temp, [299 299 3]);
                dataTrain_resized.Labels(i_train) = dataTrain.Labels(i_train);
            end

            dataTest_resized.Data = zeros(299, 299, 3, length(dataTest.Files));
            for i_test = 1:length(dataTest.Files)
                A = imread(dataTest.Files{i_test});
                data_temp = repmat(A,[1 1 3]);
                dataTest_resized.Data(:, :, :, i_test) =  imresize3(data_temp, [299 299 3]);
                dataTest_resized.Labels(i_test) = dataTest.Labels(i_test);
            end
        end

        net = [];
        for fold_i = 1:fold_num          
            switch networks_i
                case 1
                    net.Layers = [
                    imageInputLayer([224 224 3], 'Name', 'input')

                    convolution2dLayer(3,32,'Padding','same', 'Name', 'cov1')
                    batchNormalizationLayer('Name', 'batch1')
                    reluLayer('Name', 'relu1')
                    maxPooling2dLayer(2,'Stride',2 ,'Name', 'pool1')

                    convolution2dLayer(3,64,'Padding','same', 'Name', 'cov2')
                    batchNormalizationLayer('Name', 'batch2')
                    reluLayer('Name', 'relu2')
                    maxPooling2dLayer(2,'Stride',2 ,'Name', 'pool2')

                    convolution2dLayer(3,96,'Padding','same', 'Name', 'cov3')
                    batchNormalizationLayer('Name', 'batch3')
                    reluLayer('Name', 'relu3')
                    maxPooling2dLayer(2,'Stride',2 ,'Name', 'pool3')

                    convolution2dLayer(3,128,'Padding','same', 'Name', 'cov4')
                    batchNormalizationLayer('Name', 'batch4')
                    reluLayer('Name', 'relu4')
                    maxPooling2dLayer(2,'Stride',2 ,'Name', 'pool4')

                    convolution2dLayer(3,160,'Padding','same', 'Name', 'cov5')
                    batchNormalizationLayer('Name', 'batch5')
                    reluLayer('Name', 'relu5')
                    maxPooling2dLayer(2,'Stride',2 ,'Name', 'pool5')

                    convolution2dLayer(3,192,'Padding','same', 'Name', 'cov6')
                    batchNormalizationLayer('Name', 'batch6')
                    reluLayer('Name', 'relu6')
                    maxPooling2dLayer(2,'Stride',2 ,'Name', 'pool6')

                    convolution2dLayer(3,224,'Padding','same', 'Name', 'cov7')
                    batchNormalizationLayer('Name', 'batch7')
                    reluLayer('Name', 'relu7')
                    maxPooling2dLayer(2,'Stride',2 ,'Name', 'pool7')

                    fullyConnectedLayer(128 ,'Name', 'fully2')
                    dropoutLayer('Name','drop1')
                    
                    fullyConnectedLayer(numClasses,'Name','fc_final','WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
                    softmaxLayer('Name','fc_final_softmax')
                    classificationLayer('Name','ClassificationLayer_fc_final')];

%                     analyzeNetwork(layers);
                    disp('Your Own CNNs!')
                case 2
                    net= resnet18;
                    lgraph= layerGraph(net);
                    disp('CNN: ResNet18')
                case 3
                    net= resnet50;
                    lgraph= layerGraph(net);
                    disp('CNN: ResNet50')
                case 4
                    net= resnet101;
                    lgraph= layerGraph(net);
                    disp('CNN: ResNet101')
                case 5
                    net= densenet201;
                    lgraph= layerGraph(net);
                    disp('CNN: DenseNet201')
                case 6
                    net= inceptionv3;
                    lgraph= layerGraph(net);
                    disp('CNN: Inception-V3')
                case 7
                    net= inceptionresnetv2;
                    lgraph= layerGraph(net);
                    disp('CNN: Inception-ResNet-V2')
                otherwise
                    disp('Other CNNs!')
            end

            %%  1. transfer train; only train the last layer (fully connected layer)
            switch length(net.Layers)
                case 34
                    lgraph = layerGraph(net.Layers);
                case 71
                    % change the structures of last 3 layers
                    lgraph = removeLayers(lgraph,'fc1000');
                    lgraph = removeLayers(lgraph,'prob');
                    lgraph = removeLayers(lgraph,'ClassificationLayer_predictions');

                    layers_add = [
                        fullyConnectedLayer(numClasses,'Name','fc3','WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
                        softmaxLayer('Name','fc3_softmax')
                        classificationLayer('Name','ClassificationLayer_fc3')
                        ];
                    lgraph = addLayers(lgraph,layers_add);
                    lgraph = connectLayers(lgraph,'pool5','fc3');
                case 177
                    % change the structures of last 3 layers
                    lgraph = removeLayers(lgraph,'fc1000');
                    lgraph = removeLayers(lgraph,'fc1000_softmax');
                    lgraph = removeLayers(lgraph,'ClassificationLayer_fc1000');

                    layers_add = [
                        fullyConnectedLayer(numClasses,'Name','fc3','WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
                        softmaxLayer('Name','fc3_softmax')
                        classificationLayer('Name','ClassificationLayer_fc3')
                        ];
                    lgraph = addLayers(lgraph,layers_add);
                    lgraph = connectLayers(lgraph,'avg_pool','fc3');
                case 347
                    % change the structures of last 3 layers
                    lgraph = removeLayers(lgraph,'fc1000');
                    lgraph = removeLayers(lgraph,'prob');
                    lgraph = removeLayers(lgraph,'ClassificationLayer_predictions');

                    layers_add = [
                        fullyConnectedLayer(numClasses,'Name','fc3','WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
                        softmaxLayer('Name','fc3_softmax')
                        classificationLayer('Name','ClassificationLayer_fc3')
                        ];
                    lgraph = addLayers(lgraph,layers_add);
                    lgraph = connectLayers(lgraph,'pool5','fc3');
                case 708
                    % change the structures of last 3 layers
                    lgraph = removeLayers(lgraph,'fc1000');
                    lgraph = removeLayers(lgraph,'fc1000_softmax');
                    lgraph = removeLayers(lgraph,'ClassificationLayer_fc1000');

                    layers_add = [
                        fullyConnectedLayer(numClasses,'Name','fc3','WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
                        softmaxLayer('Name','fc3_softmax')
                        classificationLayer('Name','ClassificationLayer_fc3')
                        ];
                    lgraph = addLayers(lgraph,layers_add);
                    lgraph = connectLayers(lgraph,'avg_pool','fc3');
                case 315          
                    % change the structures of last 3 layers
                    lgraph = removeLayers(lgraph,'predictions');
                    lgraph = removeLayers(lgraph,'predictions_softmax');
                    lgraph = removeLayers(lgraph,'ClassificationLayer_predictions');

                    layers_add = [
                        fullyConnectedLayer(numClasses,'Name','fc3','WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
                        softmaxLayer('Name','fc3_softmax')
                        classificationLayer('Name','ClassificationLayer_fc3')
                        ];
                    lgraph = addLayers(lgraph,layers_add);
                    lgraph = connectLayers(lgraph,'avg_pool','fc3');
                case 824
                    % change the structures of last 3 layers
                    lgraph = removeLayers(lgraph,'predictions');
                    lgraph = removeLayers(lgraph,'predictions_softmax');
                    lgraph = removeLayers(lgraph,'ClassificationLayer_predictions');

                    layers_add = [
                        fullyConnectedLayer(numClasses,'Name','fc3','WeightLearnRateFactor',1,'BiasLearnRateFactor',1)
                        softmaxLayer('Name','fc3_softmax')
                        classificationLayer('Name','ClassificationLayer_fc3')
                        ];
                    lgraph = addLayers(lgraph,layers_add);
                    lgraph = connectLayers(lgraph,'avg_pool','fc3');
                otherwise
                    disp('Design CNNs By Yourself!')
            end


            miniBatchSize = 32;
            learnRate = 0.01;
            options = trainingOptions('sgdm', ...
                'InitialLearnRate',learnRate, ...
                'MaxEpochs',20, ...
                'MiniBatchSize',miniBatchSize, ...
                'Shuffle','every-epoch', ...
                'Verbose',false, ...
                'ValidationData',{dataTest_resized.Data, dataTest_resized.Labels}, ...
                'LearnRateSchedule','piecewise',...
                'LearnRateDropFactor',0.1, ...
                'LearnRateDropPeriod',8,...
                'ExecutionEnvironment','gpu',...
                'Plots','training-progress');
%             'LearnRateDropFactor',0.1, ...
                %'LearnRateDropPeriod',8, ...
% ,... 
%                 'ExecutionEnvironment','gpu'
            
            
            for train_str_i = [1 2 3 4]

                1 sgdm
                tic
                % net = trainNetwork(dataTrain_resized.Data, dataTrain_resized.Labels, lgraph, options);
                net_fuzzy = []; Fuzzy = []; info = [];
                [net_fuzzy Fuzzy info] = trainNetworkFuzzy(dataTrain_resized.Data, dataTrain_resized.Labels, lgraph, options, train_str_i);

                YPred = classify(net_fuzzy, dataTest_resized.Data);
                YValidation = dataTest_resized.Labels;
                Accuracy_table(networks_i, fold_i, train_str_i) = sum(YPred' == YValidation)/numel(YValidation);
                info_all(networks_i, fold_i, train_str_i) = {info};
                toc
            end
%             TrainingPlot(info_all, networks_i, fold_i);
                    
        end
    end

%     file_name = strcat('FP_r0.1_ra0.2_kp0.001kd46_dr', Dataset_name);
%     file_name = strcat('./YourCNN/AccuracyTable_', Dataset_name);
%     save(file_name, 'Accuracy_table');
    
    file_name = strcat('Draw_r0.01_ra0.02_kp0.001kd1000_dr3', Dataset_name);
    save(file_name, 'info_all');

%     print('Mean Values:')
%     mean(Accuracy_table, 2)
%     print('Var Values:')
%     var(Accuracy_table, 2)

end



function [dataTrain, dataTest, labels] = merchData(file_path)
 
% unzip(fullfile(matlabroot,'examples','nnet','MerchData.zip'));
 
data = imageDatastore(file_path,...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');
 
[dataTrain, dataTest] = splitEachLabel(data,0.9);
dataTrain = shuffle(dataTrain);

labels= unique(data.Labels);

% dataTrain_resized =  @(loc)imresize(imread(dataTrain(loc)), [224 224]);
% dataTest_resized =  @(loc)imresize(imread(dataTest(loc)), [224 224]);

end


function stop = SaveTrainingPlot(info)
stop = false;
if info.State == "done"
    currentfig = findall(groot,'Type','Figure');
    savefig(currentfig,'test.fig')
end
end

function TrainingPlot(info_all, networks_i, fold_i)

figure;
for train_str_i = 1:4

    info = info_all(networks_i, fold_i, train_str_i);
    switch train_str_i
        case 2
            plot(info{1}.TrainingAccuracy, 'color', [0.6350 0.0780 0.1840], 'LineWidth', 2); hold on;
        case 3
            plot(info.TrainingAccuracy, 'color', [0.9290 0.6940 0.1250], 'linewidth', 2); hold on;
        case 4
            plot(info.TrainingAccuracy, 'color', [0.4660 0.6740 0.1880], 'linewidth', 2); hold on;
        otherwise
            disp('other value');
    end
end
file_name = strcat('Network_' + num2str(networks_i) + '_' + 'Fold_' + num2str(fold_i) + '_TrainingAccuracy' );
saveas(gcf, file_name);
close all;

figure;
for train_str_i = 1:4

    info = info_all{networks_i, fold_i, train_str_i};
    switch train_str_i
        case 2
            plot(info.ValidationAccuracy, 'color', [0.6350 0.0780 0.1840], 'linewidth', 2); hold on;
        case 3
            plot(info.ValidationAccuracy, 'color', [0.9290 0.6940 0.1250], 'linewidth', 2); hold on;
        case 4
            plot(info.ValidationAccuracy, 'color', [0.4660 0.6740 0.1880], 'linewidth', 2); hold on;
        otherwise
            disp('other value');
    end
end
file_name = strcat('Network_' + num2str(networks_i) + '_' + 'Fold_' + num2str(fold_i) + '_ValidationAccuracy');
saveas(gcf, file_name);
close all;

figure;
for train_str_i = 1:4

    info = info_all{networks_i, fold_i, train_str_i};
    switch train_str_i
        case 2
            plot(info.TrainingLoss, 'color', [0.6350 0.0780 0.1840], 'linewidth', 2); hold on;
        case 3
            plot(info.TrainingLoss, 'color', [0.9290 0.6940 0.1250], 'linewidth', 2); hold on;
        case 4
            plot(info.TrainingLoss, 'color', [0.4660 0.6740 0.1880], 'linewidth', 2); hold on;
        otherwise
            disp('other value');
    end
end
file_name = strcat('Network_' + num2str(networks_i) + '_' + 'Fold_' + num2str(fold_i) + '_TrainingLoss');
saveas(gcf, file_name);
close all;

figure;
for train_str_i = 1:4

    info = info_all{networks_i, fold_i, train_str_i};
    switch train_str_i
        case 2
            plot(info.ValidationLoss, 'color', [0.6350 0.0780 0.1840], 'linewidth', 2); hold on;
        case 3
            plot(info.ValidationLoss, 'color', [0.9290 0.6940 0.1250], 'linewidth', 2); hold on;
        case 4
            plot(info.ValidationLoss, 'color', [0.4660 0.6740 0.1880], 'linewidth', 2); hold on;
        otherwise
            disp('other value');
    end
end
file_name = strcat('Network_' + num2str(networks_i) + '_' + 'Fold_' + num2str(fold_i) + '_ValidationLoss');
saveas(gcf, file_name);
close all;


end

function [alpha, beta_geometric, beta_arithmetic] = calculate_factors(x, y, E, E_C, tau_1, tau_2)
    % Calculate the contraction-expansion factor alpha for an input value x,
    % boundary E, and hereditary coefficient tau_1.
    alpha = abs(x) / E ^ tau_1;

    % Calculate the contraction-expansion factor beta for input values x, y,
    % boundaries E, E_C, and hereditary coefficients tau_1, tau_2 using
    % the geometric method.
    beta_geometric = (abs(x) / E) ^ tau_1 * (abs(y) / E_C) ^ tau_2;

    % Calculate the contraction-expansion factor beta using the arithmetic method.
    beta_arithmetic = 0.5 * ((abs(x) / E) ^ tau_1 + (abs(y) / E_C) ^ tau_2);
end

% Example usage:
x = 0.5;  % Example input value for x
y = 0.3;  % Example input value for y
E = 1.0;  % Boundary E for x
E_C = 1.0;  % Boundary E_C for y
tau_1 = 0.5;  % Example tau_1
tau_2 = -0.5;  % Example tau_2

% Calculate alpha and beta using both methods
[alpha, beta_geometric, beta_arithmetic] = calculate_factors(x, y, E, E_C, tau_1, tau_2);

% Display the results
disp(['alpha: ', num2str(alpha)]);
disp(['beta_geometric: ', num2str(beta_geometric)]);
disp(['beta_arithmetic: ', num2str(beta_arithmetic)]);
