% organizza_per_usda.m
% Sposta immagini LUCAS scaricate in cartelle per classe USDA

imgFolder = 'lucas_images';  % Cartella dove ho salvato tutte le immagini rinominate
csvFile = 'LUCAS_Text_All_10032025.csv';  % Il file CSV
outputFolder = 'dataset_lucas_usda';  % Dove spostare le immagini organizzate

% Leggi CSV
opts = detectImportOptions(csvFile);
opts = setvartype(opts, 'POINTID', 'string');
opts = setvartype(opts, 'USDA', 'string');
data = readtable(csvFile, opts);

% Filtra righe valide (senza valori POINTID e USDA mancanti)
validRows = ~(ismissing(data.POINTID) | ismissing(data.USDA));
pointMap = containers.Map(data.POINTID(validRows), data.USDA(validRows));

% Cerca tutte le immagini nella cartella
imgFiles = dir(fullfile(imgFolder, '*.jpg'));
spostate = 0;

for i = 1:numel(imgFiles)
    file = imgFiles(i).name;

    if ~endsWith(file, 'P.jpg')
        continue;  % salta immagini non "point"
    end

    % POINTID = primi 8 caratteri del nome
    pointID = extractBefore(file, 9);

    if isKey(pointMap, pointID)
        usda = pointMap(pointID);
        if usda == ""
            continue;
        end

        % Crea cartella USDA se non esiste
        classFolder = fullfile(outputFolder, matlab.lang.makeValidName(usda));
        if ~exist(classFolder, 'dir')
            mkdir(classFolder);
        end

        % Sposta immagine
        src = fullfile(imgFolder, file);
        dest = fullfile(classFolder, file);
        movefile(src, dest);
        spostate = spostate + 1;

        fprintf('%s → %s\n', file, usda);
    else
        fprintf('X  %s: POINTID %s non trovato nel CSV\n', file, pointID);
    end
end

fprintf('\nV %d immagini spostate in "%s"\n', spostate, outputFolder);



