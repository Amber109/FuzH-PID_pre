classdef Trainer < handle
    % Trainer   Trains a network
    
    %   Copyright 2015-2019 The MathWorks, Inc.
    
    properties(Access = protected)
        Options
        Schedule
        Precision
        Reporter
        SummaryFcn
        ExecutionStrategy
        StopTrainingFlag
        StopReason
        InterruptException
    end
    
    methods
        function this = Trainer( ...
                opts, precision, reporter, executionSettings, summaryFcn)
            % Trainer    Constructor for a network trainer
            %
            % opts - training options (nnet.cnn.TrainingOptionsSGDM)
            % precision - data precision
            this.Options = opts;
            scheduleArguments = iGetArgumentsForScheduleCreation(opts.LearnRateScheduleSettings);
            this.Schedule = nnet.internal.cnn.LearnRateScheduleFactory.create(scheduleArguments{:});
            this.Precision = precision;
            this.Reporter = reporter;
            this.SummaryFcn = summaryFcn;
            % Declare execution strategy
            if ismember( executionSettings.executionEnvironment, {'gpu'} )
                this.ExecutionStrategy = nnet.internal.cnn.TrainerGPUStrategy;
            else
                this.ExecutionStrategy = nnet.internal.cnn.TrainerHostStrategy;
            end
            
            % Print execution environment if in verbose mode
            iPrintExecutionEnvironment(opts, executionSettings);
            
            % Register a listener to detect requests to terminate training
            addlistener( reporter, ...
                'TrainingInterruptEvent', @this.stopTrainingCallback);
        end
        
        function net = initializeNetwork(this, net, data)
            % Perform any initialization steps required by the input and
            % output layers
            
            % Store any classification labels or response names in the
            % appropriate output layers.
            net = net.storeResponseData(data.ResponseMetaData);
            
            if this.Options.ResetInputNormalization
                net = net.resetNetworkInitialization();
            end
            
            % Setup reporters
            willInputNormalizationBeComputed = net.hasEmptyInputStatistics();
            reporterSetupInfo = nnet.internal.cnn.util.ReporterSetupInfo(willInputNormalizationBeComputed);
            this.Reporter.setup(reporterSetupInfo);
            
            if willInputNormalizationBeComputed
                % Always use 'truncateLast' as we want to process all the data we have.
                savedEndOfEpoch = data.EndOfEpoch;
                data.EndOfEpoch = 'truncateLast';
                
                % Compute statistics on host or in distributed environment
                stats = iCreateStatisticsAccumulator(net.InputSizes,this.Precision);
                stats = this.computeInputStatisticsInEnvironment(data,stats,net);
                
                % Initialize input layers with statistics
                net = net.initializeNetwork(stats);
                
                data.EndOfEpoch = savedEndOfEpoch;
            end
        end
        
        function net = train(this, net, data)
            % train   Train a network
            %
            % Inputs
            %    net -- network to train
            %    data -- data encapsulated in a data dispatcher
            % Outputs
            %    net -- trained network
            reporter = this.Reporter;
            schedule = this.Schedule;
            prms = collectSettings(this, net);
            data.start();
            summary = this.SummaryFcn(data, prms.maxEpochs);
            
            regularizer = iCreateRegularizer('l2',net.LearnableParameters,this.Precision,this.Options);
            
            solver = iCreateSolver(net.LearnableParameters,this.Precision,this.Options);
            
            trainingTimer = tic;
            
            reporter.start();
            iteration = 0;
            this.StopTrainingFlag = false;
            this.StopReason = nnet.internal.cnn.util.TrainingStopReason.FinalIteration;
            gradientThresholdOptions = struct('Method', this.Options.GradientThresholdMethod,...
                'Threshold', this.Options.GradientThreshold);
            gradThresholder = nnet.internal.cnn.GradientThresholder(gradientThresholdOptions);
            learnRate = initializeLearning(this);
            
            for epoch = 1:prms.maxEpochs
                this.shuffle( data, prms.shuffleOption, epoch );
                data.start();
                while ~data.IsDone && ~this.StopTrainingFlag
                    [X, response] = data.next();
                    % Cast data to appropriate execution environment for
                    % training and apply transforms
                    X = this.ExecutionStrategy.environment(X);
                    response = this.ExecutionStrategy.environment(response);
                    
                    propagateState = iNeedsToPropagateState(data);
                    [gradients, predictions, states] = this.computeGradients(net, X, response, propagateState);
                    
                    % Reuse the layers outputs to compute loss
                    miniBatchLoss = net.loss( predictions, response );
                    
                    gradients = regularizer.regularizeGradients(gradients,net.LearnableParameters);
                    
                    gradients = thresholdGradients(gradThresholder,gradients);
                    
                    velocity = solver.calculateUpdate(gradients,learnRate);
                    
                    net = net.updateLearnableParameters(velocity);
                    
                    net = net.updateNetworkState(states);
                    
                    elapsedTime = toc(trainingTimer);
                    
                    iteration = iteration + 1;
                    summary.update(predictions, response, epoch, iteration, elapsedTime, miniBatchLoss, learnRate, data.IsDone);
                    % It is important that computeIteration is called
                    % before reportIteration, so that the summary is
                    % correctly updated before being reported
                    reporter.computeIteration( summary, net );
                    reporter.reportIteration( summary );
                end
                learnRate = schedule.update(learnRate, epoch);
                
                reporter.reportEpoch( epoch, iteration, net );
                
                % If an interrupt request has been made, break out of the
                % epoch loop
                if this.StopTrainingFlag
                    break;
                end
            end
            reporter.computeFinish( summary, net );
            reporter.finish( summary, this.StopReason );
        end
        
        function [net Fuzzy] = trainFuzzy(this, net, data, Flag_Fuzzy)
            % train   Train a network
            %
            % Inputs
            %    net -- network to train
            %    data -- data encapsulated in a data dispatcher
            % Outputs
            %    net -- trained network
            reporter = this.Reporter;
            schedule = this.Schedule;
            prms = collectSettings(this, net);
            data.start();
            summary = this.SummaryFcn(data, prms.maxEpochs);
            
            regularizer = iCreateRegularizer('l2',net.LearnableParameters,this.Precision,this.Options);
            
            solver = iCreateSolver(net.LearnableParameters,this.Precision,this.Options);
            
            trainingTimer = tic;
            
            reporter.start();
            iteration = 0;
            this.StopTrainingFlag = false;
            this.StopReason = nnet.internal.cnn.util.TrainingStopReason.FinalIteration;
            gradientThresholdOptions = struct('Method', this.Options.GradientThresholdMethod,...
                'Threshold', this.Options.GradientThreshold);
            gradThresholder = nnet.internal.cnn.GradientThresholder(gradientThresholdOptions);
            learnRate = initializeLearning(this);
            
            %% ************** Set fuzzy control  **************
            Num_iteration = 1;%FLAG
            Fuzzy.WeightFuzzy = cell(15,Num_iteration);
            Fuzzy.activationsBufferFuzzy = cell(1,15);
                        
            for epoch = 1:prms.maxEpochs
                this.shuffle( data, prms.shuffleOption, epoch );
                data.start();
                while ~data.IsDone && ~this.StopTrainingFlag
                    [X, response] = data.next();
                    % Cast data to appropriate execution environment for
                    % training and apply transforms
                    X = this.ExecutionStrategy.environment(X);
                    response = this.ExecutionStrategy.environment(response);
                    
                    %%  *******************************************************
                   
                    Num_Cov = 0;
                    for Num_layers = 1:length(net.Layers)
                        Name_layers = net.Layers{Num_layers, 1}.Name;
                        if (contains(Name_layers,'cov'))
                            Num_Cov = Num_Cov + 1;
                        end
                    end
                    e_1 = cell(Num_Cov-1, Num_iteration);      
                    Ee = cell(Num_Cov-1, Num_iteration);     
                    r = cell(Num_Cov-1, Num_iteration);
                    r_initial = cell(Num_Cov-1, Num_iteration);
                    e = cell(Num_Cov-1, Num_iteration);
                    
                    % according to the paper "PID Controller-Based
                    % Stochastic Optimization Acceleration for Deep Neural Networks",
                    % we set Kp, Kd and Ki as follow, the subsequent Equations using is (5-27,5-28,5-29):
                    kp = 0.001;
                    ki = 0.001;
                    kd = 10;
                    %----------------------------------------------------------------
                    if Flag_Fuzzy == 4
                        FuzzyPIDFIS = readfis('/home/fuzzyPID_optimization/Fuzzy_CNN/FuzzyPID_range0.001');
                        for i_iterate = 1:Num_iteration
                            Num_Cov = 0;
                            propagateState = iNeedsToPropagateState(data);
                            [gradients, activationsBufferFuzzy, predictions, states] = this.computeGradients(net, X, response, propagateState);
                            
                            % Reuse the layers outputs to compute loss, with the use of Equation (5-34)
                            miniBatchLoss = net.loss( predictions, response );
                            
                            gradients = regularizer.regularizeGradients(gradients,net.LearnableParameters);
                            
                            gradients = thresholdGradients(gradThresholder,gradients);
                            
                            velocity = solver.calculateUpdate(gradients,learnRate);
                            Fuzzy.activationsBufferFuzzy(i_iterate,epoch) = {activationsBufferFuzzy};
                            
                            %                             WeightFuzzy_ec = [];with the use of Equation (5-30-5-33)
                            Fuzzy.WeightFuzzy(i_iterate,epoch) = {squeeze(net.Layers{6,1}.Weights.Value(:,:,:,:))};
                            
                            %                             WeightFuzzy_ec(:,:,:) = velocity{5}(:,:,:,1);
                            %                             Size_WeightFuzzy = size(WeightFuzzy_e);
                            %                             WeightFuzzy_u = zeros(Size_WeightFuzzy);
                            
                            for Num_layers = 1:length(net.Layers)
                                Name_layers = net.Layers{Num_layers, 1}.Name;
                                if (Num_layers>3) & (contains(Name_layers,'cov'))
                                    if i_iterate == 5
                                        ha = 1;
                                    end
                                    Num_Cov = Num_Cov + 1;
                                    WeightFuzzy_e = [];
                                    WeightFuzzy_e = {net.Layers{Num_layers,1}.Weights.Value(:,:,:,:)};
                                %% Fuzzy Control,with the use of Equation (5-35-5-46)
                                    r(Num_Cov, i_iterate) = WeightFuzzy_e;      %����ֵ
                                    if i_iterate == 1
                                        Size_r = size(r{Num_Cov, i_iterate});
                                        r_initial(Num_Cov, i_iterate) = {zeros(Size_r)};%��ʼ����
                                        e_1(Num_Cov, i_iterate) = {zeros(Size_r)};
                                        Ee(Num_Cov, i_iterate) = e_1(Num_Cov, i_iterate);
                                    else if i_iterate ~= 1
                                            r_initial(Num_Cov, i_iterate) = r(Num_Cov, i_iterate-1);%��ʼ����
                                            e_1(Num_Cov, i_iterate) = e(Num_Cov, i_iterate-1);
                                            Ee(Num_Cov, i_iterate) = Ee(Num_Cov, i_iterate-1);
                                        end
                                    end
                                    e(Num_Cov, i_iterate) = {r{Num_Cov, i_iterate}-r_initial{Num_Cov, i_iterate}};   %����ź�
                                    
                                    pid_P = []; pid_D = []; pid_I = [];
                                    pid_D = gather(kp*(e{Num_Cov, i_iterate} - e_1{Num_Cov, i_iterate}));
                                    pid_P = gather(ki*(e{Num_Cov, i_iterate}));
                                    pid_I = gather(kd*Ee{Num_Cov, i_iterate});
                                    Size_WeightFuzzy = size(WeightFuzzy_e{1});
                                    WeightFuzzy_u = zeros(Size_WeightFuzzy);
                                    for i=1:1:Size_WeightFuzzy(1)
                                        for j=1:1:Size_WeightFuzzy(2)
                                            for k=1:1:Size_WeightFuzzy(3)
                                                for p = 1:Size_WeightFuzzy(4)
                                                    WeightFuzzy_u(i, j, k, p) = evalfis(FuzzyPIDFIS, [pid_P(i, j, k, p) pid_D(i, j, k, p)]);
                                                end
                                            end
                                        end
                                    end
                                    Ee(Num_Cov, i_iterate) = {Ee{Num_Cov, i_iterate} + e{Num_Cov, i_iterate}};    %�����ۼӺ�
                                    e_1(Num_Cov, i_iterate) = {e{Num_Cov, i_iterate}};		%ǰһ������źŵ�ֵ
                                    
                                    net.Layers{Num_layers,1}.Weights.Value(:,:,:,:) = net.Layers{Num_layers,1}.Weights.Value(:,:,:,:) + gpuArray(WeightFuzzy_u);
                                end
                            end
                            net = net.updateLearnableParameters(velocity);
                        end
                    elseif Flag_Fuzzy == 3
                        for i_iterate = 1:Num_iteration
                            Num_Cov = 0;
                            propagateState = iNeedsToPropagateState(data);
                            [gradients, activationsBufferFuzzy, predictions, states] = this.computeGradients(net, X, response, propagateState);
                            
                            % Reuse the layers outputs to compute loss
                            miniBatchLoss = net.loss( predictions, response );
                            
                            gradients = regularizer.regularizeGradients(gradients,net.LearnableParameters);
                            
                            gradients = thresholdGradients(gradThresholder,gradients);
                            
                            velocity = solver.calculateUpdate(gradients,learnRate);
                            Fuzzy.activationsBufferFuzzy(i_iterate,epoch) = {activationsBufferFuzzy};
                            Fuzzy.WeightFuzzy(i_iterate,epoch) = {squeeze(net.Layers{6,1}.Weights.Value(:,:,:,:))};
                            
                            for Num_layers = 1:length(net.Layers)
                                Name_layers = net.Layers{Num_layers, 1}.Name;
                                if (Num_layers>3) & (contains(Name_layers,'cov'))
                                    if i_iterate == 5
                                        ha = 1;
                                    end
                                    Num_Cov = Num_Cov + 1;
                                    WeightFuzzy_e = [];
                                    WeightFuzzy_e = {net.Layers{Num_layers,1}.Weights.Value(:,:,:,:)};
                                %% PID Control
                                    r(Num_Cov, i_iterate) = WeightFuzzy_e;      %����ֵ
                                    if i_iterate == 1
                                        Size_r = size(r{Num_Cov, i_iterate});
                                        r_initial(Num_Cov, i_iterate) = {zeros(Size_r)};%��ʼ����
                                        e_1(Num_Cov, i_iterate) = {zeros(Size_r)};
                                        Ee(Num_Cov, i_iterate) = e_1(Num_Cov, i_iterate);
                                    else if i_iterate ~= 1
                                            r_initial(Num_Cov, i_iterate) = r(Num_Cov, i_iterate-1);%��ʼ����
                                            e_1(Num_Cov, i_iterate) = e(Num_Cov, i_iterate-1);
                                            Ee(Num_Cov, i_iterate) = Ee(Num_Cov, i_iterate-1);
                                        end
                                    end
                                    e(Num_Cov, i_iterate) = {r{Num_Cov, i_iterate}-r_initial{Num_Cov, i_iterate}};   %����ź�
                                    
                                    pid_P = []; pid_D = []; pid_I = [];
                                    pid_D = kp*(e{Num_Cov, i_iterate} - e_1{Num_Cov, i_iterate});
                                    pid_P = ki*(e{Num_Cov, i_iterate});
                                    pid_I = kd*Ee{Num_Cov, i_iterate};
                                    Size_WeightFuzzy = size(WeightFuzzy_e);
                                    WeightFuzzy_u = pid_P + pid_I + pid_D;

                                    Ee(Num_Cov, i_iterate) = {Ee{Num_Cov, i_iterate} + e{Num_Cov, i_iterate}};    %�����ۼӺ�
                                    e_1(Num_Cov, i_iterate) = {e{Num_Cov, i_iterate}};		%ǰһ������źŵ�ֵ
                                    
                                    net.Layers{Num_layers,1}.Weights.Value(:,:,:,:) = net.Layers{Num_layers,1}.Weights.Value(:,:,:,:) + WeightFuzzy_u;
                                end
                            end
                            net = net.updateLearnableParameters(velocity);
                        end
                    else if Flag_Fuzzy == 1
                            propagateState = iNeedsToPropagateState(data);
                            [gradients, activationsBufferFuzzy, predictions, states] = this.computeGradients(net, X, response, propagateState);
                            
                            Fuzzy.activationsBufferFuzzy(epoch) = {activationsBufferFuzzy};
                            Fuzzy.WeightFuzzy(epoch) = {squeeze(net.Layers{6,1}.Weights.Value(:,:,:,:))};
                            
                            % Reuse the layers outputs to compute loss
                            miniBatchLoss = net.loss( predictions, response );
                            
                            gradients = regularizer.regularizeGradients(gradients,net.LearnableParameters);
                            
                            gradients = thresholdGradients(gradThresholder,gradients);
                            
                            velocity = solver.calculateUpdate(gradients,learnRate);
                            
                            net = net.updateLearnableParameters(velocity);
                        else if Flag_Fuzzy == 2
                                for i_iterate = 1:Num_iteration
                                    propagateState = iNeedsToPropagateState(data);
                                    [gradients, activationsBufferFuzzy, predictions, states] = this.computeGradients(net, X, response, propagateState);
                                    
                                    Fuzzy.activationsBufferFuzzy(i_iterate,epoch) = {activationsBufferFuzzy};
                                    Fuzzy.WeightFuzzy(i_iterate,epoch) = {squeeze(net.Layers{6,1}.Weights.Value(:,:,:,:))};
                                    
                                    % Reuse the layers outputs to compute loss
                                    miniBatchLoss = net.loss( predictions, response );
                                    
                                    gradients = regularizer.regularizeGradients(gradients,net.LearnableParameters);
                                    
                                    gradients = thresholdGradients(gradThresholder,gradients);
                                    
                                    velocity = solver.calculateUpdate(gradients,learnRate);
                                    
                                    net = net.updateLearnableParameters(velocity);
                                end
                            end
                        end
                    end
                    
                    net = net.updateNetworkState(states);
                    elapsedTime = toc(trainingTimer);
                    
                    iteration = iteration + 1;
                    summary.update(predictions, response, epoch, iteration, elapsedTime, miniBatchLoss, learnRate, data.IsDone);
                    % It is important that computeIteration is called
                    % before reportIteration, so that the summary is
                    % correctly updated before being reported
                    reporter.computeIteration( summary, net );
                    reporter.reportIteration( summary );
                end
                learnRate = schedule.update(learnRate, epoch);
                
                reporter.reportEpoch( epoch, iteration, net );
                
                % If an interrupt request has been made, break out of the
                % epoch loop
                if this.StopTrainingFlag
                    break;
                end
            end
            reporter.computeFinish( summary, net );
            reporter.finish( summary, this.StopReason );
        end
        
        function net = finalizeNetwork(this, net, data)
            % Perform any finalization steps required by the layers
            
            % Always use 'truncateLast' as we want to process all the data we have.
            savedEndOfEpoch = data.EndOfEpoch;
            data.EndOfEpoch = 'truncateLast';
            
            % Call shared implementation
            net = this.doFinalize(net, data);
            
            data.EndOfEpoch = savedEndOfEpoch;
        end
    end
    
    methods(Access = protected)
        function stopTrainingCallback(this, ~, evtData)
            % stopTraining  Callback triggered by interrupt events that
            % want to request training to stop
            this.StopTrainingFlag = true;
            this.StopReason = evtData.TrainingStopReason;
        end
        
        function settings = collectSettings(this, net)
            % collectSettings  Collect together fixed settings from the
            % Trainer and the data and put in the correct form.
            settings.maxEpochs = this.Options.MaxEpochs;
            settings.lossFunctionType = iGetLossFunctionType(net);
            settings.shuffleOption = this.Options.Shuffle;
            settings.numOutputs = numel(net.OutputSizes);
            settings.InputObservationDim = cellfun(@(sz)numel(sz)+1, net.InputSizes, 'UniformOutput', false);
            settings.OutputObservationDim = cellfun(@(sz)numel(sz)+1, net.OutputSizes, 'UniformOutput', false);
        end
        
        function learnRate = initializeLearning(this)
            % initializeLearning  Set initial learning rate.
            learnRate = this.Precision.cast( this.Options.InitialLearnRate );
        end
        
        function [gradients, activationsBufferFuzzy, predictions, states] = computeGradients(~, net, X, Y, propagateState)
            % computeGradients   Compute the gradients of the network. This
            % function returns also the network output so that we will not
            % need to perform the forward propagation step again.
            [gradients, activationsBufferFuzzy, predictions, states] = net.computeGradientsForTraining(X, Y, propagateState);
        end
        
        function stats = computeInputStatisticsInEnvironment(this, data, stats, net )
            % Initialization steps running on the host
            stats = doComputeStatistics(this, data, stats, net);
        end
        
        function stats = doComputeStatistics(this, data, stats, net)
            % Compute statistics from the training data set
            % Do one epoch
            data.start();
            while ~data.IsDone
                X = data.next();
                if ~isempty(X) % In the parallel case X can be empty
                    % Cast data to appropriate execution environment
                    X = this.ExecutionStrategy.environment(X);
                    X = iWrapInCell(X);
                    
                    % Accumulate statistics for each input
                    for i = 1:numel(stats)
                        % The ImageInputLayer might have data augmentation
                        % transforms that have to be applied before
                        % computing statistics (e.g. cropping).
                        if isa(net.InputLayers{i},'nnet.internal.cnn.layer.ImageInput')
                            X{i} = apply(net.InputLayers{i}.TrainTransforms,X{i});
                        end
                        
                        stats{i} = accumulate(stats{i}, X{i});
                    end
                end
            end
        end
        
        function net = doFinalize(this, net, data)
            % Perform any finalization steps required by the layers
            isFinalizable = @(l)isa(l,'nnet.internal.cnn.layer.Finalizable') && ...
                l.NeedsFinalize;
            needsFinalize = cellfun(isFinalizable, net.Layers);
            if any(needsFinalize)
                % Do one final epoch
                data.start();
                while ~data.IsDone
                    X = data.next();
                    if ~isempty(X) % In the parallel case X can be empty
                        % Cast data to appropriate execution environment for
                        % training and apply transforms
                        X = this.ExecutionStrategy.environment(X);
                        
                        % Ask the network to finalize
                        net = finalizeNetwork(net, X);
                    end
                end
                
            end
        end
    end
    
    methods(Access = protected)
        function shuffle(~, data, shuffleOption, epoch)
            % shuffle   Shuffle the data as per training options
            if ~isequal(shuffleOption, 'never') && ...
                    ( epoch == 1 || isequal(shuffleOption, 'every-epoch') )
                data.shuffle();
            end
        end
    end
end

function regularizer = iCreateRegularizer(name,learnableParameters,precision,regularizationOptions)
regularizer = nnet.internal.cnn.regularizer.RegularizerFactory.create(name,learnableParameters,precision,regularizationOptions);
end

function solver = iCreateSolver(learnableParameters,precision,trainingOptions)
solver = nnet.internal.cnn.solver.SolverFactory.create(learnableParameters,precision,trainingOptions);
end

function stats = iCreateStatisticsAccumulator(inputSizes,outputType)
stats = nnet.internal.cnn.statistics.AccumulatorFactory.create(inputSizes,outputType);
end

function tf = iNeedsToPropagateState(data)
tf = isa(data,'nnet.internal.cnn.SequenceDispatcher') && data.IsNextMiniBatchSameObs;
end

function t = iGetLossFunctionType(net)
if isempty(net.Layers)
    t = 'nnet.internal.cnn.layer.NullLayer';
else
    t = class(net.Layers{end});
end
end

function scheduleArguments = iGetArgumentsForScheduleCreation(learnRateScheduleSettings)
scheduleArguments = struct2cell(learnRateScheduleSettings);
end

function iPrintMessage(messageID, varargin)
string = getString(message(messageID, varargin{:}));
fprintf( '%s\n', string );
end

function iPrintExecutionEnvironment(opts, executionSettings)
% Print execution environment if in 'auto' mode
if opts.Verbose
    if ismember(opts.ExecutionEnvironment, {'auto'})
        if ismember(executionSettings.executionEnvironment, {'cpu'})
            iPrintMessage( ...
                'nnet_cnn:internal:cnn:Trainer:TrainingInSerialOnCPU');
        elseif ismember(executionSettings.executionEnvironment, {'gpu'})
            iPrintMessage( ...
                'nnet_cnn:internal:cnn:Trainer:TrainingInSerialOnGPU');
        end
    elseif ismember(opts.ExecutionEnvironment, {'parallel'})
        if ismember(executionSettings.executionEnvironment, {'cpu'})
            iPrintMessage( ...
                'nnet_cnn:internal:cnn:Trainer:TrainingInParallelOnCPUs');
        elseif ismember(executionSettings.executionEnvironment, {'gpu'})
            iPrintMessage( ...
                'nnet_cnn:internal:cnn:Trainer:TrainingInParallelOnGPUs');
        end
    end
end
end

function X = iWrapInCell(X)
if ~iscell(X)
    X = {X};
end
end
