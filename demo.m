clc
close all

%% 
addpath('data'); addpath('functions'); 
Files = dir(fullfile('data', '*.mat'));
Max_datanum = length(Files);

%% 
for data_num = 1:Max_datanum   
    Dname = Files(data_num).name;
    disp(['***********The test data name is: ***' num2str(data_num) '***'  Dname '****************'])
    load(Dname);
    dname = Dname(1:end-4);
    
    k = 10; p = [0.1,0.3,0.5,0.7,0.9,1,1.1,1.3,1.5,1.7,1.9]; 
    th_value = [1,2,3]; viewnum = length(X); v_n_value = 1:1:viewnum;
    for p_i = 1:length(p)
        p_value = p(p_i);
        
       %% 
        file_path = 'Results/';
        folder_name = ['p=' num2str(p_value)];  
        file_path_name = strcat(file_path,folder_name);
        if exist(file_path_name,'dir') == 0   
           mkdir(file_path_name);
        end
        file_mat_path = [file_path_name '/'];
        
        for th_i = 1:length(th_value)
            th = th_value(th_i);
            for v_n_i = 1:length(v_n_value)
                v_n = v_n_value(v_n_i);
            
                tic
                [Z_PKN,~,~,~] = APGM(X,Y,0,k,p_value,th,v_n);
                time_APGM = toc;
                APGM_result = ClusteringMeasure(Y,Z_PKN);
      
                file_name = [dname '_th_' num2str(th) '_v_' num2str(v_n) '.mat'];
                save ([file_mat_path,file_name],'time_APGM','APGM_result');
                                
            end
        end                      
    end
end  