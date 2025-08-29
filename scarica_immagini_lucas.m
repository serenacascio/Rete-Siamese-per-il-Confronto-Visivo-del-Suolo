% scarica_immagini_lucas.m
% URL base dove si trovano le immagini del dataset LUCAS 
% ho usato italia 2018, 2015,con range 405:504
% spagna 2018 con range 155:382
base_url = 'https://gisco-services.ec.europa.eu/lucas/photos/2018/ES/';
main_folders = 155:382;
subfolder_nums = 0:999;
save_folder = 'lucas_images'; % Cartella locale dove salvare tutte le immagini

% Se la cartella locale non esiste, creala
if ~exist(save_folder, 'dir')
    mkdir(save_folder);
end

parpool("local");  % Avvia il pool di processi paralleli 

% Inizio del ciclo parallelo: ogni iterazione del parfor lavora su una cartella principale diversa
parfor idx = 1:length(main_folders)
    i = main_folders(idx);
    main_folder = sprintf('%d/', i);
    
    % Ciclo interno
    for j = subfolder_nums
        sub_folder = sprintf('%03d/', j);
        folder_url = [base_url main_folder sub_folder];

        % Prova a leggere il contenuto HTML, se la cartella non esiste passa alla successiva
        try
            options = weboptions('Timeout', 10, 'ContentType', 'text');
            html = webread(folder_url, options);
        catch
            continue
        end
        
        % Cerca immagini con lettere C, E, N, P, S, W
        %expr = '(?<=href=")(\d{8}[CENPSW]\.jpg)"';
         % Cerca immagini con lettera P utile per il nostro esempio
         expr = '(?<=href=")(\d{8}[P]\.jpg)"';
        tokens = regexp(html, expr, 'tokens');
        
         % Se non ci sono immagini, passa alla sottocartella successiva
        if isempty(tokens)
            continue
        end

        %local_folder = fullfile(save_folder, main_folder, sub_folder);
        %if ~exist(local_folder, 'dir')
        %    mkdir(local_folder); 
       % end

       % Ciclo su tutte le immagini trovate
        for k = 1:length(tokens)
            img_name = tokens{k}{1};
            img_url = [folder_url img_name];
            local_path = fullfile(save_folder, img_name);

            if exist(local_path, 'file')
                continue;  % Salta se già scaricata
            end

            try
                websave(local_path, img_url);
                fprintf('V %s\n', img_name);
            catch
                fprintf('X Errore: %s\n', img_url);
            end
        end
    end
end

delete(gcp('nocreate'));  % Chiudi il pool a fine lavoro
