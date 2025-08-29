% genera_coppie_siamese.m 
% Crea un file CSV con coppie simili (label 1) e dissimili (label 0)

datasetDir = 'dataset_lucas_usda';
outputCSV = 'coppie_siamese.csv';

% Leggi tutte le classi (cartelle)
classDirs = dir(datasetDir);
classDirs = classDirs([classDirs.isdir] & ~startsWith({classDirs.name}, '.'));

pairs = {};
labels = [];

% === COPPIE SIMILI (label = 1)
for i = 1:numel(classDirs)
    className = classDirs(i).name;
    files = dir(fullfile(datasetDir, className, '*.jpg'));
    fileNames = {files.name};

    % Genera tutte le combinazioni di 2 immagini
    for j = 1:numel(fileNames)
        for k = j+1:numel(fileNames)
            img1 = fullfile(className, fileNames{j});
            img2 = fullfile(className, fileNames{k});
            pairs{end+1, 1} = img1;
            pairs{end, 2} = img2;
            labels(end+1) = 1;
        end
    end
end

% === COPPIE DISSIMILI (label = 0)
for i = 1:numel(classDirs)
    for j = i+1:numel(classDirs)
        class1 = classDirs(i).name;
        class2 = classDirs(j).name;

        files1 = dir(fullfile(datasetDir, class1, '*.jpg'));
        files2 = dir(fullfile(datasetDir, class2, '*.jpg'));

        n = min(numel(files1), numel(files2));
        n = min(n, 30);  % limitiamo le coppie dissimili

        for k = 1:n
            img1 = fullfile(class1, files1(k).name);
            img2 = fullfile(class2, files2(k).name);
            pairs{end+1, 1} = img1;
            pairs{end, 2} = img2;
            labels(end+1) = 0;
        end
    end
end

% Salva in CSV
T = table(pairs(:,1), pairs(:,2), labels', 'VariableNames', {'image1', 'image2', 'label'});
writetable(T, outputCSV);

fprintf('\nFile "%s" creato con %d coppie (simili + dissimili)\n', outputCSV, height(T));
