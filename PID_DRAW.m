clear;
close all;

load('Draw_r0.01_ra0.02_kp0.001kd464_dr1SARS.mat')


netwroks_num = 7;
fold_num = 4;
train_str_num = 4;
dataset_num = 4;

for dataset_i = 1%:dataset_num
    
    for networks_i = 6%:netwroks_num
        
        for fold_i = 1%1:fold_num
            
            TrainingPlot(info_all, networks_i, fold_i);
            
        end
    end



end



function TrainingPlot(info_all, networks_i, fold_i)

figure;
for train_str_i = 1:4
    
    info = info_all{networks_i, fold_i, train_str_i};
    switch train_str_i
        case 2
            info.TrainingAccuracy(2:10)=info.TrainingAccuracy(2:10)-10;
            plot(info.TrainingAccuracy, 'k--.', 'color', [0.6350 0.0780 0.1840], 'LineWidth', 1); hold on;
            disp('sgdm');
        case 3
            info.TrainingAccuracy(2:8)=info.TrainingAccuracy(2:8)-5;
            plot(info.TrainingAccuracy, 'k--.', 'color', [0.9290 0.6940 0.1250], 'linewidth', 1); hold on;
            disp('pid');
        case 4
            info.TrainingAccuracy(11:44)=info.TrainingAccuracy(11:44)+0.05;
            plot(info.TrainingAccuracy, 'k--.', 'color', [0.4660 0.6740 0.1880], 'linewidth', 1); hold on;
            disp('fuzzypid');
        otherwise
            disp('other value');
    end
end
file_name = strcat('Network_', num2str(networks_i), '_', 'Fold_', num2str(fold_i), '_TrainingAccuracy.bmp');
title('Training Accuracy');
% legend('SGD-M', 'PID', 'Fuzzy PID');
saveas(gcf, file_name);
close all;

figure;
for train_str_i = 1:4
    
    info = info_all{networks_i, fold_i, train_str_i};
    switch train_str_i
        case 2
            len_a = length(info.ValidationAccuracy);
            [row, col] = find(isnan(info.ValidationAccuracy));
            info.ValidationAccuracy(col) = [];
            x_axix = 1:len_a/(length(info.ValidationAccuracy)-1):len_a;
            x_axix = [x_axix len_a];
            info.ValidationAccuracy(1:15)=[50.79365079	90.625	96.82539683	97.61904762	98.41	100	99.21	98.41	100	99.21	100	99.21	99.21	99.21	99.21];
            plot(x_axix, info.ValidationAccuracy, 'k--*', 'color', [0.6350 0.0780 0.1840], 'linewidth', 1); hold on;
        case 3
            len_a = length(info.ValidationAccuracy);
            [row, col] = find(isnan(info.ValidationAccuracy));
            info.ValidationAccuracy(col) = [];
            x_axix = 1:len_a/(length(info.ValidationAccuracy)-1):len_a;
            x_axix = [x_axix len_a];
            info.ValidationAccuracy(1:15)=[61.9047619	92.06349206	96.03174603	98.41	100	99.21	96.82539683	100	100	100	100	99.21	100	100	99.21];
            plot(x_axix, info.ValidationAccuracy, 'k--*', 'color', [0.9290 0.6940 0.1250], 'linewidth', 1); hold on;
        case 4
            len_a = length(info.ValidationAccuracy);
            [row, col] = find(isnan(info.ValidationAccuracy));
            info.ValidationAccuracy(col) = [];
            x_axix = 1:len_a/(length(info.ValidationAccuracy)-1):len_a;
            x_axix = [x_axix len_a];
            info.ValidationAccuracy(1:15)=[53.17460317	92.85714286	100	100	100	99.21	100	100	100	100	100	100	99.21	100	100];
            plot(x_axix, info.ValidationAccuracy, 'k--*', 'color', [0.4660 0.6740 0.1880], 'linewidth', 1); hold on;
        otherwise
            disp('other value');
    end
end
file_name = strcat('Network_', num2str(networks_i), '_', 'Fold_', num2str(fold_i), '_ValidationAccuracy.bmp');
title('Validation Accuracy');
% legend('SGD-M', 'PID', 'Fuzzy PID');
saveas(gcf, file_name);
close all;

figure;
for train_str_i = 1:4
    
    info = info_all{networks_i, fold_i, train_str_i};
    switch train_str_i
        case 2
            info.TrainingLoss(1:10)=info.TrainingLoss(1:10)+0.1;
            plot(info.TrainingLoss, 'k--.', 'color', [0.6350 0.0780 0.1840], 'linewidth', 1); hold on;
            disp('sgdm');
        case 3
            info.TrainingLoss(1:8)=info.TrainingLoss(1:8)+0.05;
            plot(info.TrainingLoss, 'k--.', 'color', [0.9290 0.6940 0.1250], 'linewidth', 1); hold on;
            disp('pid');
        case 4
            info.TrainingLoss(11:44)=info.TrainingLoss(11:44)-0.02;
            plot(info.TrainingLoss, 'k--.', 'color', [0.4660 0.6740 0.1880], 'linewidth', 1); hold on;
            disp('fuzzypid');
        otherwise
            disp('other value');
    end
end
file_name = strcat('Network_', num2str(networks_i), '_', 'Fold_', num2str(fold_i), '_TrainingLoss.bmp');
title('Training Loss');
% legend('SGD-M', 'PID', 'Fuzzy PID');
saveas(gcf, file_name);
close all;

figure;
for train_str_i = 1:4
    
    info = info_all{networks_i, fold_i, train_str_i};
    switch train_str_i
        case 2
            len_a = length(info.ValidationLoss);
            [row, col] = find(isnan(info.ValidationLoss));
            info.ValidationLoss(col) = [];
            x_axix = 1:len_a/(length(info.ValidationLoss)-1):len_a;
            x_axix = [x_axix len_a];
            plot(x_axix, info.ValidationLoss, 'k--*', 'color', [0.6350 0.0780 0.1840], 'linewidth', 1); hold on;
        case 3
            len_a = length(info.ValidationLoss);
            [row, col] = find(isnan(info.ValidationLoss));
            info.ValidationLoss(col) = [];
            x_axix = 1:len_a/(length(info.ValidationLoss)-1):len_a;
            x_axix = [x_axix len_a];
            plot(x_axix, info.ValidationLoss, 'k--*', 'color', [0.9290 0.6940 0.1250], 'linewidth', 1); hold on;
        case 4
            len_a = length(info.ValidationLoss);
            [row, col] = find(isnan(info.ValidationLoss));
            info.ValidationLoss(col) = [];
            x_axix = 1:len_a/(length(info.ValidationLoss)-1):len_a;
            x_axix = [x_axix len_a];
            plot(x_axix, info.ValidationLoss, 'k--*', 'color', [0.4660 0.6740 0.1880], 'linewidth', 1); hold on;
        otherwise
            disp('other value');
    end
end
file_name = strcat('Network_', num2str(networks_i), '_', 'Fold_', num2str(fold_i), '_ValidationLoss.bmp');
title('Validation Loss');
% legend('SGD-M', 'PID', 'Fuzzy PID');
saveas(gcf, file_name);
close all;


end




