% Define the lists
Ins = {"Cello", "Piano", "Organ"};
Inter = {"C3ASharp3", "C3CSharp3", "C3FSharp3", "C3G3", ...
         "C4ASharp4", "C4CSharp4", "C4FSharp4", "C4G4"};

% Initialize cell arrays to store filenames
PEMO_List = {};

% Nested loops to generate filenames
for i = 1:length(Ins)
    for j = 1:length(Inter)
        instrument = Ins{i};
        interval = Inter{j};
        
        Ref_filename = sprintf("stimulus/ReGen/%s_%s.wav", instrument, interval);
        Ref_32k_filename = sprintf("stimulus/ReGen/%s_%s_32kbps.wav", instrument, interval);
        Ref_96k_filename = sprintf("stimulus/ReGen/%s_%s_96kbps.wav", instrument, interval);
        Ak_filename = sprintf("stimulus/ReGen/%s_%s_EQ_acoustic_enhance.wav", instrument, interval);
        Ak_96k_filename = sprintf("stimulus/ReGen/%s_%s_EQ_acoustic_enhance_96kbps.wav", instrument, interval);
        Mk_filename = sprintf("stimulus/ReGen/%s_%s_EQ_musical_enhance.wav", instrument, interval);
        Mk_96k_filename = sprintf("stimulus/ReGen/%s_%s_EQ_musical_enhance_96kbps.wav", instrument, interval);
        fprintf("Comparing the %s in %s\n", instrument, interval);
        [ref_data,ref_fs] = audioread(Ref_filename);
        [ref_data_32k,ref_fs] = audioread(Ref_32k_filename);
        [ref_data_96k,ref_fs] = audioread(Ref_96k_filename);
        [ak_data,ak_fs] = audioread(Ak_filename);
        [ak_data_96k,ak_fs] = audioread(Ak_96k_filename);
        [mk_data,ak_fs] = audioread(Mk_filename);
        [mk_data_96k,ak_fs] = audioread(Mk_96k_filename);
        [ak_PSM, ak_PSMt, ak_ODG_PEMO, ak_PSM_inst] = audioqual(ref_data,ak_data,ref_fs);
        [mk_PSM, mk_PSMt, mk_ODG_PEMO, mk_PSM_inst] = audioqual(ref_data,mk_data,ref_fs);

        % Store filenames in cell arrays
        [ref_data_32k_PSM, ref_data_32k_PSMt, ref_data_32k_ODG_PEMO, ref_data_32k_PSM_inst] = audioqual(ref_data,ref_data_32k,ref_fs);
        PEMO_List{end+1} = ref_data_32k_ODG_PEMO;
        [ref_data_96k_PSM, ref_data_96k_PSMt, ref_data_96k_ODG_PEMO, ref_data_96k_PSM_inst] = audioqual(ref_data,ref_data_96k,ref_fs);
        PEMO_List{end+1} = ref_data_96k_ODG_PEMO;
        [ak_data_96k_PSM, ak_data_96k_PSMt, ak_data_96k_ODG_PEMO, ak_data_96k_PSM_inst] = audioqual(ref_data,ak_data_96k,ref_fs);
        PEMO_List{end+1} = ak_data_96k_ODG_PEMO;
        [mk_data_96k_PSM, mk_data_96k_PSMt, mk_data_96k_ODG_PEMO, mk_data_96k_PSM_inst] = audioqual(ref_data,mk_data_96k,ref_fs);
        PEMO_List{end+1} = mk_data_96k_ODG_PEMO;
        PEMO_List{end+1} = 0;
        [ak_data_PSM, ak_data_PSMt, ak_data_ODG_PEMO, ak_data_PSM_inst] = audioqual(ref_data,ak_data,ref_fs);
        PEMO_List{end+1} = ak_data_ODG_PEMO;
        [mk_data_PSM, mk_data_PSMt, mk_data_ODG_PEMO, mk_data_PSM_inst] = audioqual(ref_data,mk_data,ref_fs);
        PEMO_List{end+1} = mk_data_ODG_PEMO;
        % Print comparison message
        % fprintf("result is %s",ak_PSMt,mk_PSMt);  
    end
end

PEMO_List = PEMO_List';


% Define column headers
headers = {'PEMO'};

% Write to CSV file
csv_filename = 'PEMO_ReGen_Data.csv';
fid = fopen(csv_filename, 'w');  % Open file for writing

% Write column headers
fprintf(fid, '%s,%s\n', headers{:});

% Write data values
fclose(fid);  % Close the file
writecell(PEMO_List, csv_filename, 'WriteMode', 'append');  % Append the data below headers
