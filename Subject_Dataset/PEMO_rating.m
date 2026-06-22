% Define the lists
Ins = {"Cello", "Piano", "Organ"};
Inter = {"C3ASharp3", "C3CSharp3", "C3FSharp3", "C3G3", ...
         "C4ASharp4", "C4CSharp4", "C4FSharp4", "C4G4"};

% Initialize cell arrays to store filenames
Ak_List = {};
Mk_List = {};

% Nested loops to generate filenames
for i = 1:length(Ins)
    for j = 1:length(Inter)
        instrument = Ins{i};
        interval = Inter{j};
        
        Ref_filename = sprintf("stimulus/%s_%s_96kbps_lame_CBR.wav", instrument, interval);
        Ak_filename = sprintf("stimulus/%s_%s_EQ_acoustic_enhance_96kbps_lame_CBR.wav", instrument, interval);
        Mk_filename = sprintf("stimulus/%s_%s_EQ_musical_enhance_96kbps_lame_CBR.wav", instrument, interval);
        fprintf("Comparing the %s in %s\n", instrument, interval);
        [ref_data,ref_fs] = audioread(Ref_filename);
        [ak_data,ak_fs] = audioread(Ak_filename);
        [mk_data,ak_fs] = audioread(Mk_filename);

        [ak_PSM, ak_PSMt, ak_ODG_PEMO, ak_PSM_inst] = audioqual(ref_data,ak_data,ref_fs);
        [mk_PSM, mk_PSMt, mk_ODG_PEMO, mk_PSM_inst] = audioqual(ref_data,mk_data,ref_fs);

        % Store filenames in cell arrays
        Ak_List{end+1} = ak_PSMt;
        Mk_List{end+1} = mk_PSMt;
        
        % Print comparison message
        fprintf("AK result is %s and Mk_result is %s",ak_PSMt,mk_PSMt);  
    end
end

Ak_List = Ak_List';
Mk_List = Mk_List';
data = [Ak_List, Mk_List];

% Define column headers
headers = {'PEMO_AK', 'PEMO_MK'};

% Write to CSV file
csv_filename = 'PEMO_Data.csv';
fid = fopen(csv_filename, 'w');  % Open file for writing

% Write column headers
fprintf(fid, '%s,%s\n', headers{:});

% Write data values
fclose(fid);  % Close the file
writecell(data, csv_filename, 'WriteMode', 'append');  % Append the data below headers
