% pulizia_coppie_lucas.m
% Script MATLAB per la pre-pulizia delle coppie di immagini LUCAS
% filtriamo il CSV per rimuovere coppie con immagini mancanti o errate

%% Parametri iniziali
imageFolder = 'dataset_lucas_usda_resized';        % Cartella contenente le immagini resizeate
inputFile = 'coppie_siamese.csv';                 % File CSV originale
outputFile = 'coppie_siamese_filtrate.csv';       % File CSV filtrato (output)

%% Lettura CSV
data = readtable(inputFile);
n = height(data);
validRows = false(n,1);  % Inizializziamo la maschera per i record validi

%% Controllo validità immagini con parfor 
parfor i = 1:n
    try
        img1Path = fullfile(imageFolder, data.image1{i});
        img2Path = fullfile(imageFolder, data.image2{i});
        if isfile(img1Path) && isfile(img2Path)
            validRows(i) = true;
        end
    catch
        validRows(i) = false;
    end
end

%% Scrittura nuovo CSV solo con righe valide
filtrato = data(validRows, :);
fprintf('Coppie valide trovate: %d su %d\n', height(filtrato), n);
writetable(filtrato, outputFile);
fprintf('File salvato: %s\n', outputFile);

