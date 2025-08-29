% resize_lucas_images.m
% Script per ridimensionare le immagini LUCAS con crop centrale

inputDir = 'dataset_lucas_usda';         % Cartella originale immagini
outputDir = 'dataset_lucas_usda_resized'; % Cartella di salvataggio immagini ridimensionate
resizeTo = [300 300];              % Dimensione target (es. 300x300)

% Crea la cartella di output se non esiste
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Scorri tutte le sottocartelle
classFolders = dir(fullfile(inputDir, '*'));
classFolders = classFolders([classFolders.isdir] & ~startsWith({classFolders.name}, '.'));

totalSaved = 0;

for i = 1:length(classFolders)
    className = classFolders(i).name;
    inputClassDir = fullfile(inputDir, className);
    outputClassDir = fullfile(outputDir, className);
    if ~exist(outputClassDir, 'dir')
        mkdir(outputClassDir);
    end

    % Estensioni multiple (lo facciamo solo per sicurezza)
    imageFiles = [ ...
        dir(fullfile(inputClassDir, '*.jpg')) ; ...
        dir(fullfile(inputClassDir, '*.jpeg')) ; ...
        dir(fullfile(inputClassDir, '*.png')) ...
    ];

    for j = 1:length(imageFiles)
        imgName = imageFiles(j).name;
        imgPath = fullfile(inputClassDir, imgName);

        fprintf("Leggo: %s\n", imgPath);

        try
            img = imread(imgPath);
            imgCropped = centerCropImage(img);
            imgResized = imresize(imgCropped, resizeTo);

            % Salva immagine
            outPath = fullfile(outputClassDir, imgName);
            imwrite(imgResized, outPath);
            totalSaved = totalSaved + 1;
        catch ME
            fprintf("Errore con %s: %s\n", imgPath, ME.message);
        end
    end

    fprintf("Classe '%s': %d immagini elaborate\n", className, length(imageFiles));
end

fprintf("\nCompletato! %d immagini totali ridimensionate e salvate in '%s'\n", totalSaved, outputDir);


% Funzione per crop centrale
function cropped = centerCropImage(img)
    sz = size(img);
    h = sz(1); w = sz(2);
    minDim = min(h, w);
    y = floor((h - minDim)/2) + 1;
    x = floor((w - minDim)/2) + 1;
    cropped = imcrop(img, [x y minDim-1 minDim-1]);
end
