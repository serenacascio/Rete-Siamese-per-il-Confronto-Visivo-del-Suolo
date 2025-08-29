% siamese_train_lucas.m
% Script MATLAB per l'addestramento di una rete Siamese sul dataset LUCAS

%% Parametri iniziali
imageFolder = 'dataset_lucas_usda_resized';  % Cartella con immagini resizeate
coppieFile = 'coppie_siamese_filtrate.csv';  % Coppie valide pre-filtrate
imageSize = [300 300 3];                     % Dimensione immagini compatibile con la rete

%% Caricamento coppie
coppie = readtable(coppieFile);
fileList = cell(height(coppie), 1);
for i = 1:height(coppie)
    fileList{i} = struct(...
        'image1', fullfile(imageFolder, coppie.image1{i}), ...
        'image2', fullfile(imageFolder, coppie.image2{i}), ...
        'label', coppie.label(i));
end

%% Definizione rete Siamese
layers = [
    imageInputLayer(imageSize, 'Name','input')

    convolution2dLayer(3,8,'Padding','same','Name','conv1')
    batchNormalizationLayer('Name','bn1')
    reluLayer('Name','relu1')
    maxPooling2dLayer(2,'Stride',2,'Name','pool1')

    convolution2dLayer(3,16,'Padding','same','Name','conv2')
    batchNormalizationLayer('Name','bn2')
    reluLayer('Name','relu2')

    % Appiattisce spazialmente a 1x1 con una media globale
    globalAveragePooling2dLayer('Name','gap')

    % Embedding finale (64-D)
    fullyConnectedLayer(64,'Name','fc')
];
dlgraph = layerGraph(layers);
snet = dlnetwork(dlgraph);


%% Parametri di training
miniBatchSize = 128;  % Aumentato da 32 a 128 per maggiore efficienza computazionale
epochs = 5;
learningRate = 0.001;
idx = randperm(length(fileList));

%% Ciclo di addestramento
for epoch = 1:epochs
    lossTotal = 0;
    for i = 1:miniBatchSize:length(fileList)
        batchStart = tic;  % Timer per misurare i tempi per batch
        batch = fileList(idx(i:min(i+miniBatchSize-1,end)));

        % Preallocazione batch
        X1cell = cell(1, numel(batch));
        X2cell = cell(1, numel(batch));
        Ycell = zeros(1, numel(batch), 'single');

        % Precaricamento con controllo validità
        parfor j = 1:numel(batch)
            try
                img1 = im2double(imread(batch{j}.image1));
                img2 = im2double(imread(batch{j}.image2));

                if isequal(size(img1), imageSize) && isequal(size(img2), imageSize)
                    X1cell{j} = img1;
                    X2cell{j} = img2;
                    Ycell(j) = single(batch{j}.label);
                else
                    X1cell{j} = [];
                    X2cell{j} = [];
                    Ycell(j) = NaN;
                end
            catch
                X1cell{j} = [];
                X2cell{j} = [];
                Ycell(j) = NaN;
            end
        end

        % Rimozione voci invalide
        validIdx = ~cellfun(@isempty, X1cell) & ~isnan(Ycell);
        X1cell = X1cell(validIdx);
        X2cell = X2cell(validIdx);
        Ycell = Ycell(validIdx);

        if isempty(X1cell)
            continue;
        end

        % Preparazione mini-batch
        [X1, X2, Y] = preprocessMiniBatch(X1cell, X2cell, Ycell, imageSize);

        % Passaggio forward e backward
        [loss, gradients] = dlfeval(@modelLoss, snet, X1, X2, Y);

        % Aggiornamento pesi (senza ricreare la rete)
        snet = updateNetworkWeights(snet, gradients, learningRate);

        % Somma della loss
        lossTotal = lossTotal + double(gather(extractdata(loss)));
        fprintf("  Batch %d/%d - Tempo: %.2fs\n", i, length(fileList), toc(batchStart));  % Log tempo batch
    end
    fprintf("Epoch %d - Loss media: %.4f\n", epoch, lossTotal);
end

%% Salvataggio del modello addestrato
save('snet_lucas_final.mat','snet');

%% Funzioni di supporto
function [X1, X2, Y] = preprocessMiniBatch(X1cell, X2cell, Yvec, imageSize)
    X1 = dlarray(cat(4, X1cell{:}), 'SSCB');
    X2 = dlarray(cat(4, X2cell{:}), 'SSCB');
    Y = dlarray(reshape(single(Yvec), 1, 1, 1, []), 'SSCB');
end

function [loss, gradients] = modelLoss(net, X1, X2, Y)
    Z1 = forward(net, X1);   % dlarray, dimensione [64 1 1 B] o [64 B]
    Z2 = forward(net, X2);

    % Rende esplicitamente vettoriali gli embedding per ogni elemento del batch
    Z1 = reshape(Z1, [], size(Z1, ndims(Z1))); % [64, B]
    Z2 = reshape(Z2, [], size(Z2, ndims(Z2))); % [64, B]

    % L2-normalization per stabilizzare la scala delle distanze
    Z1 = Z1 ./ (sqrt(sum(Z1.^2,1)) + eps);
    Z2 = Z2 ./ (sqrt(sum(Z2.^2,1)) + eps);

    % Distanza euclidea per elemento del batch -> [1, B]
    D = sqrt(sum((Z1 - Z2).^2, 1));

    % Y deve essere [1, B] per il broadcasting corretto
    if ~isequal(size(Y,1),1)
        Y = reshape(Y, 1, []);
    end

    % Loss contrastiva semplice (0=simili, 1=diversi)
    loss = mean((1 - Y) .* (1 - exp(-D)) + Y .* (D.^2));

    % Gradiente rispetto ai learnables
    gradients = dlgradient(loss, net.Learnables.Value);
end


function net = updateNetworkWeights(net, gradients, learnRate)
    learnablesTbl = net.Learnables;
    for i = 1:height(learnablesTbl)
        learnablesTbl.Value{i} = learnablesTbl.Value{i} - learnRate * gradients{i};
    end
    net.Learnables = learnablesTbl;
end

