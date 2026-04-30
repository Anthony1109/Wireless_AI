%% Demo model to run the COST 2100 channel model (Modified for Exercise 2.15)
% This script automatically generates multiple channel datasets by
% varying the spatial distributions of Mobile Stations (MSPos). 
% These datasets are saved as .mat files and used to train and test 
% the generalization capabilities of the CsiNet autoencoder.

% -------------------------------------------------------------------------
% 1. Global Simulation Parameters
% -------------------------------------------------------------------------
Network = 'Indoor_CloselySpacedUser_2_6GHz'; % Indoor propagation environment
Link = 'Multiple';                           % Multi-user MIMO scenario
Antenna = 'MIMO_Cyl_patch';                  % Using Cylindrical patch array at the Base Station
Band = 'Wideband';                           % Wideband frequency settings

% Define the total number of distinct datasets to generate 
% (Required: more than five for the generalization experiment)
num_datasets = 6; 

for dataset_idx = 1:num_datasets
    fprintf('\n======================================================\n');
    fprintf('Generating Dataset %d of %d...\n', dataset_idx, num_datasets);
    fprintf('======================================================\n');
    
    switch Network
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        case 'Indoor_CloselySpacedUser_2_6GHz'
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            scenario = 'LOS'; % Line-of-Sight environment
            freq = [2.58e9 2.62e9]; % Starting and ending frequencies [Hz]
            snapNum = 50; % Number of channel snapshots (time instances)
            snapRate = 50; % Number of snapshots per second (determines Doppler)
            
            % Base coordinates for closely-spaced users [x, y, z] in meters
            % Represents a cluster of 9 users in a specific indoor location
            MSPos_base  = [  -2.5600    1.7300    2.2300;...
                             -3.0800    1.7300    2.2300;...
                             -2.5600    2.6200    2.5800;...
                             -4.6400    1.7300    2.2300;...
                             -2.5600    4.4000    3.3000;...
                             -3.0800    3.5100    2.9400;...
                             -3.6000    4.4000    3.3000;...
                             -4.1200    4.4000    3.3000;...
                             -4.1200    2.6200    2.5800]; 
                         
            % -------------------------------------------------------------
            % [CORE MODIFICATION] Spatial Diversity Injection
            % Alter the physical distribution of users based on the current loop index
            % -------------------------------------------------------------
            switch dataset_idx
                case 1
                    % Dataset 1: The original, unmodified base distribution
                    MSPos = MSPos_base; 
                case 2
                    % Dataset 2: Translate the entire user cluster +5 meters along the X-axis
                    MSPos = MSPos_base + repmat([5, 0, 0], 9, 1); 
                case 3
                    % Dataset 3: Translate the entire user cluster +5 meters along the Y-axis
                    MSPos = MSPos_base + repmat([0, 5, 0], 9, 1); 
                case 4
                    % Dataset 4: Spatial Expansion (Scale by 1.5)
                    % Increases the distance between users, making the cluster sparser
                    MSPos = MSPos_base * 1.5; 
                case 5
                    % Dataset 5: Spatial Contraction (Scale by 0.5)
                    % Decreases the distance between users, creating an extreme hotspot
                    MSPos = MSPos_base * 0.5; 
                case 6
                    % Dataset 6: Random Uniform Distribution
                    % Users are scattered randomly within a 10m x 10m area
                    MSPos = [rand(9,1)*10-5, rand(9,1)*10-5, ones(9,1)*2.2]; 
            end
            
            % Mobile Station velocity vector [x, y, z] in m/s
            MSVelo = repmat([-.25,0,0],9,1); 
            
            % Initial Base Station array center position
            BSPosCenter  = [0.30 -4.37 3.20]; 
            BSPosSpacing = [0 0 0]; 
            BSPosNum = 1; 
            
            % Normalize coordinate system: 
            % Adjust BS position relative to the mean center of the MS cluster
            BSPosCenter = BSPosCenter - mean(MSPos); 
            MSPos = MSPos - repmat(mean(MSPos),size(MSPos,1),1);
            
        otherwise
            error('This script is currently optimized exclusively for the Indoor_CloselySpacedUser_2_6GHz network.');
    end
      
    %% Execute COST 2100 Channel Model
    % Extracts Multipath Components (MPCs) based on the physical environment 
    % and the dynamic geometry of the Tx/Rx positions.
    [...
        paraEx,...       
        paraSt,...       
        link,...         
        env...           
    ] = cost2100...
    (...
        Network,...      
        scenario,...     
        freq,...         
        snapRate,...     
        snapNum,...      
        BSPosCenter,...  
        BSPosSpacing,... 
        BSPosNum,...     
        MSPos,...        
        MSVelo...        
        );         

    %% Environment Visualization (Disabled for batch processing)
    if 0  % Set to 1 to enable plotting, kept at 0 to prevent freezing during automated dataset generation
        switch Network
            case {'IndoorHall_5GHz','SemiUrban_300MHz'}   
                 visual_channel(paraEx, paraSt, link, env);
            case {'SemiUrban_VLA_2_6GHz','SemiUrban_CloselySpacedUser_2_6GHz','Indoor_CloselySpacedUser_2_6GHz'}   
                 visualize_channel_env(paraEx, paraSt, link, env); axis equal; view(2);
        end   
    end

    %% Combine Propagation Data with Antenna Patterns
    switch Antenna
        %%%%%%%%%%%%%%%%%%%%%%%
        case 'MIMO_Cyl_patch' % 128-element cylindrical array at the Base Station
        %%%%%%%%%%%%%%%%%%%%%%%   
            USE_EADF = 1; % Flag to use Effective Antenna Degree of Freedom (EADF)
        
            if (USE_EADF)
                BSantEADF = load('BS_Cyl_EADF.mat','F'); % Load BS antenna EADF data
            else
                BSantPattern = load('BS_Cyl_AntPattern.mat'); % Load standard BS antenna pattern
            end
            
            Nbs_ant = 128; % Total number of BS antennas
            Nms = size(MSPos, 1); % Total number of MS users (9)
       
            MSantPattern = load('MS_AntPattern_User.mat'); % MS antenna pattern including user body effect
            
            % Calculate frequency bin spacing
            delta_f = (freq(2)-freq(1))/256;
            
            % Generate Impulse Response (Time Domain)
            % Output dimension: [snapshots, delays, MS_users, BS_antennas]
            if (USE_EADF)
                ir_Cyl_Patch = create_IR_Cyl_EADF(link, freq, delta_f, BSantEADF.F, MSantPattern);
            else
                ir_Cyl_Patch = create_IR_Cyl(link, freq, delta_f, BSantPattern, MSantPattern);
            end
            
            % Convert Impulse Response to Channel Transfer Function (Frequency Domain)
            % Fast Fourier Transform across the delay dimension
            % Output dimension: [snapshots, freq_bins, MS_users, BS_antennas]
            H_transfer = fft(ir_Cyl_Patch, [], 2);
            
            % --- Plotting functions disabled to avoid interrupting the batch loop ---
            % figure, mesh(...)
            % figure, plot(...)
            % figure, cdfplot(...)
            % ------------------------------------------------------------------------
            
        otherwise
            error('Please use MIMO_Cyl_patch or implement specific saving logic for other antenna types.');
    end
    
    %% Export Dataset for Python / TensorFlow (CsiNet)
    % Save the generated frequency-domain channel matrix (H_transfer) and spatial data 
    % as a .mat file for neural network training and evaluation.
    filename = sprintf('channel_dataset_%d.mat', dataset_idx);
    save(filename, 'H_transfer', 'MSPos', 'BSPosCenter');
    fprintf('>> Successfully saved: %s\n', filename);
end

fprintf('\nAll %d datasets have been generated successfully!\n', num_datasets);