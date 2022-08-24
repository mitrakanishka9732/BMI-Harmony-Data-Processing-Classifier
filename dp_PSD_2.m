clear all
close all
clc
%% Load data

addpath(genpath('./functions'));
[data, hdr] = sload(['./test_data/Sub_03_trialRun_1_wHarmony.gdf']); 

labels=hdr.EVENT.POS(:,:);
data = data(1:labels(end),:);

%% Delete Channels 
clc
new_data = data(:,1:32); 

%removing specific channels: T7, T8, M1, M2, FP1, FPZ, FP2, O1, Oz, O2
% Keep [F7, F8, P7, P8], remove when feature selection 
new_data(:,[1,2,3,13,14,18,19,30,31,32]) = [];

%updating channel label list 
chLabel = hdr.Label(1:32,:); 
chLabel([1,2,3,13,14,18,19,30,31,32],:) = []; 

%load and update topoplot map for only 22 channels 
load('ch32Locations.mat'); 
ch32Locations(:,[1,2,3,13,14,18,19,30,31,32]) = [];




%% Bandpass/Notch Filter the data

%Orset et al., 2021 
fc1 = 0.01; % first cutoff frequency in Hz 
fc2 = 100; % second cutoff frequency in Hz
fs = hdr.SampleRate; 

% normalize the frequencies
Wp = [fc1 fc2]*2/fs;
[b,a]=butter(2,Wp,'bandpass');

%filter data - user filter, instead of filtfit, because real-time
%processing
filter_sig_1 = filtfilt(b,a,new_data); 


%60HZ notch filter 
wo = 60/(512/2);  
bw = wo/35;
[c,d] = iirnotch(wo,bw);

%notch filter 
filter_sig_2 = filtfilt(c,d,filter_sig_1); 

%% EPOCH EXTRACTION

%1 sec window
wsize = 1; 
%62.5msec overlap
hop = 0.0625; 

%extract labels
labels=hdr.EVENT.POS(:,:);

%trial counter: number of epochs per class * 20 
x_trial = 1;
y_trial = 1; 
z_trial = 1; 

%iterate through the whole session 
for i = 2:4:78

    %get total number of epoch in Begin MI
    num_epoch = (floor((((labels(i+2) - labels(i+1))/512) - wsize)/hop)+1);
    
    %Begin MI pos
    lab_x = labels(i+1);
    %iterate through trial
    for j = 1:1:num_epoch
        epochs_bm(:,:,x_trial) = (filter_sig_2(lab_x:lab_x+511,:));
        lab_x = lab_x + 32;    %iterate lab_x pos by hop size*512 = 32sam
        x_trial = x_trial + 1;
    end

    %END MI
    %shift window by 900ms, 0.9*512=461 
    lab_y = labels(i+2)-461; 
    %1900ms(total slide)/62.5ms = 30 epochs per trial  
    for j = 1:1:30      
        epochs_em(:,:,y_trial) = (filter_sig_2(lab_y:lab_y+511,:));
        lab_y = lab_y + 32;    %iterate lab_x pos by hop size*512 = 32sam
        y_trial = y_trial + 1;
    end

    %Rest
    %rest label position 
    lab_z = labels(i+3); 
    %same number of epochs as begin mi 
    for j = 1:1:num_epoch
        epochs_rst(:,:,z_trial) = (filter_sig_2(lab_z:lab_z+511,:));
        lab_z = lab_z + 32;    %iterate lab pos by hop size*512 = 32sam
        z_trial = z_trial+1; 
    end
    
    
end


%% PSD features 

wsize = 1;  %resolution 
fs = hdr.SampleRate;
tot_sze = size(epochs_bm,3)+size(epochs_em,3)+size(epochs_rst,3); 
bm_sze = size(epochs_bm,3); 
em_sze = size(epochs_em,3);
rst_sze = size(epochs_rst,3);

PSD_epoch(tot_sze,507) = 0;

%temp PSD features 
PSD_bm_temp(23,22) = 0; 
PSD_em_temp(23,22) = 0; 
PSD_rst_temp(23,22) = 0; 

%total PSD counter 
cnt = 1; 
%begin MI 
for i = 1:1:bm_sze
    signalOfInterest = epochs_bm(:,:,i);
    [SOIf, freq]=pwelch(signalOfInterest,wsize*fs, 0.5*wsize*fs, [], fs); 
    PSD_bm_temp = SOIf(9:31,:);
    PSD_epoch(cnt,1:506) = PSD_bm_temp(:)'; 
    PSD_epoch(cnt,507) = 1; 
    PSD_bm_temp = 0;
    cnt = cnt + 1; 
end


%end MI
for i = 1:1:em_sze
    signalOfInterest = epochs_em(:,:,i);
    [SOIf, freq]=pwelch(signalOfInterest,wsize*fs, 0.5*wsize*fs, [], fs); 
    PSD_em_temp = SOIf(9:31,:);
    PSD_epoch(cnt,1:506) = PSD_em_temp(:)'; 
    PSD_epoch(cnt,507) = 2; 
    PSD_em_temp = 0;
    cnt = cnt + 1; 
end

%rest
for i = 1:1:rst_sze
    signalOfInterest = epochs_rst(:,:,i);
    [SOIf, freq]=pwelch(signalOfInterest,wsize*fs, 0.5*wsize*fs, [], fs); 
    PSD_rst_temp = SOIf(9:31,:);
    PSD_epoch(cnt,1:506) = PSD_rst_temp(:)'; 
    PSD_epoch(cnt,507) = 3; 
    PSD_rst_temp = 0;
    cnt = cnt + 1; 
end

freq = freq(9:31,:); 

%% Fisher Score 

%class 1 vs rest
fish_score_1(1,506) = 0;
%class 2 vs rest
fish_score_2(1,506) = 0; 
%class 1 vs class 2
fish_score_3(1,506) = 0; 

PSD_bm_a=0;
PSD_bm_v=0;

PSD_em_a=0;
PSD_em_v=0;

PSD_rst_a=0;
PSD_rst_v=0; 


%iterate through all freq/channel
for i = 1:1:506
    PSD_bm_a = mean(PSD_epoch(1:bm_sze,i));
    PSD_bm_v = var(PSD_epoch(1:bm_sze,i));

    PSD_em_a = mean(PSD_epoch(bm_sze+1:(bm_sze+em_sze),i));
    PSD_em_v = var(PSD_epoch(bm_sze+1:(bm_sze+em_sze),i));

    PSD_rst_a = mean(PSD_epoch((bm_sze+em_sze)+1:(bm_sze+em_sze+rst_sze),i));
    PSD_rst_v = var(PSD_epoch((bm_sze+em_sze)+1:(bm_sze+em_sze+rst_sze),i));

    %Begin MI vs. Rest 
    abav1 = abs(PSD_bm_a - PSD_rst_a); 
    totvar1 = sqrt(PSD_bm_v^2 + PSD_rst_v^2);
    fish_score_1(1,i) = abav1/totvar1; 
    
    %END MI vs. Rest 
    abav2 = abs(PSD_em_a - PSD_rst_a); 
    totvar2 = sqrt(PSD_em_v^2 + PSD_rst_v^2);
    fish_score_2(1,i) = abav2/totvar2; 

    %Begin MI vs. END MI
    abav3 = abs(PSD_bm_a - PSD_em_a); 
    totvar3 = sqrt(PSD_bm_v^2 + PSD_em_v^2);
    fish_score_3(1,i) = abav3/totvar3; 

end 

fish_score_1_dl(23,22) = 0; 
fish_score_2_dl(23,22) = 0; 
fish_score_3_dl(23,22) = 0; 
a = 1; 

for i =1:23:506
    fish_score_1_dl(:,a) = fish_score_1(1,i:i+22)'; 
    fish_score_2_dl(:,a) = fish_score_2(1,i:i+22)'; 
    fish_score_3_dl(:,a) = fish_score_3(1,i:i+22)'; 
    a=a+1; 
end 

%histogram(fish_score_1); 


%% Top 10 Features - Begin MI/Rest

Nmax = 10; % get Nmax biggest entries
[ Avec, Ind ] = sort(fish_score_1(:),1,'descend');
max_values = Avec(1:Nmax);
[ ind_row, ind_col ] = ind2sub(size(fish_score_1),Ind(1:Nmax)); % fetch indices

elec_idx = fix((ind_col(:)-1)/23); 
frq_idx = ind_col(:) - 23*(elec_idx); 

disp("Begin MI/Rest: Top Ten Features")
disp("Band: " + (freq(frq_idx,1)) +"Hz, Channel: "+ ...
    chLabel(elec_idx+1,1) + '(' +ind_col(:)+')'+" Fisher Value: " + max_values(:))


%% Top 10 Features - END MI/Rest

Nmax = 10; % get Nmax biggest entries
[ Avec, Ind ] = sort(fish_score_2(:),1,'descend');
max_values1 = Avec(1:Nmax);
[ ind_row1, ind_col1 ] = ind2sub(size(fish_score_2),Ind(1:Nmax)); % fetch indices

elec_idx1 = fix((ind_col1(:)-1)/23); 
frq_idx1 = ind_col1(:) - 23*(elec_idx1); 

disp("END MI/Rest: Top Ten Features")
disp("Band: " + (freq(frq_idx1,1)) +"Hz, Channel: "+ ...
    chLabel(elec_idx1+1,1) + '(' +ind_col1(:)+')'+" Fisher Value: " + max_values1(:))


%% Top 10 Features - Begin MI/END MI

Nmax = 10; % get Nmax biggest entries
[ Avec, Ind ] = sort(fish_score_3(:),1,'descend');
max_values2 = Avec(1:Nmax);
[ ind_row2, ind_col2 ] = ind2sub(size(fish_score_3),Ind(1:Nmax)); % fetch indices

elec_idx2 = fix((ind_col2(:)-1)/23); 
frq_idx2 = ind_col2(:) - 23*(elec_idx2); 

disp("Begin MI/End MI: Top Ten Features")
disp("Band: " + (freq(frq_idx2,1)) +"Hz, Channel: "+ ...
    chLabel(elec_idx2+1,1) + '(' +ind_col2(:)+')'+" Fisher Value: " + max_values2(:))


%% Extract top PSD features

n_max = 5; 
PSD_bm_fs(bm_sze, n_max) = 0; 
PSD_bm_rst_fs(rst_sze, n_max) = 0; 

PSD_em_fs(em_sze, n_max) = 0; 
PSD_em_rst_fs(rst_sze, n_max) = 0; 

PSD_bm_em(bm_sze, n_max) = 0; 
PSD_em_bm(em_sze, n_max) = 0; 

%extracting the PSD from the average signals 
for i = 1:1:n_max
    PSD_bm_fs(:,i) = PSD_epoch(1:bm_sze,ind_col(i)); 
    PSD_bm_rst_fs(:,i) = PSD_epoch((bm_sze+em_sze)+1:(bm_sze+em_sze+rst_sze),ind_col(i));

    PSD_em_fs(:,i) = PSD_epoch(bm_sze+1:(bm_sze+em_sze),ind_col1(i)); 
    PSD_em_rst_fs(:,i) = PSD_epoch((bm_sze+em_sze)+1:(bm_sze+em_sze+rst_sze),ind_col1(i));
    
    PSD_bm_em(:, i) = PSD_epoch(1:bm_sze,ind_col(i)); 
    PSD_em_bm(:, i) = PSD_epoch(bm_sze+1:(bm_sze+em_sze),ind_col1(i)); 

end 


%% Saving Data

save('dp_PSDfeature_2.mat','PSD_bm_fs','PSD_bm_rst_fs','PSD_em_fs','PSD_em_rst_fs','PSD_bm_em','PSD_em_bm')

%% Plot topo
figure('units','normalized','Position',[0.1,0.1,0.5,0.5])

A = sum(fish_score_1_dl);
B = sum(fish_score_2_dl);
C = sum(fish_score_3_dl);

tiledlayout(3,1)
title('')
nexttile
x1 = topoplot(A,ch32Locations);
title('Onset: Begin MI vs. Rest')
colorbar
nexttile
x3 = topoplot(C,ch32Locations);
title('Termination: Begin MI vs. End MI')
colorbar
nexttile
x2 = topoplot(B,ch32Locations);
title('End MI vs. Rest')
colorbar

sgtitle('Subject 3 with Harmony: Topoplots')

