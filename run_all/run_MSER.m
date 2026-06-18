function run_MSER(bucketPath)
% Process benign and malignant inputs separately
inputDirs = {fullfile(bucketPath,'benign'), fullfile(bucketPath,'malignant')};
outputRoot = bucketPath;

for d = 1:length(inputDirs)
    inputDir = inputDirs{d};
    [~, categoryName] = fileparts(inputDir);

    % Define separate output folders for MSER and MSER+CLAHE
    outputDirMSER   = fullfile(outputRoot, ['output_MSER_' categoryName]);
    outputDirCLAHE  = fullfile(outputRoot, ['output_MSER_CLAHE_' categoryName]);
    if ~exist(outputDirMSER,'dir'), mkdir(outputDirMSER); end
    if ~exist(outputDirCLAHE,'dir'), mkdir(outputDirCLAHE); end

    files = dir(fullfile(inputDir,'*.png'));
    files = files(~contains({files.name},{'_MSER','_FA'})); % avoid recursion

    T_mser = table; T_mser_clahe = table;

    for idx = 1:length(files)
        currentImage = fullfile(inputDir, files(idx).name);
        [~, baseName, ~] = fileparts(currentImage);
        img = imread(currentImage);
        gray = im2gray(img);

        %% --- MSER segmentation ---
        regions = detectMSERFeatures(gray,'RegionAreaRange',[30 14000],'ThresholdDelta',2);
        mask = false(size(gray));
        for r = 1:length(regions)
            pts = regions(r).PixelList;
            mask(sub2ind(size(gray), pts(:,2), pts(:,1))) = true;
        end
        vis = repmat(gray,1,1,3);
        vis(:,:,1) = vis(:,:,1) + uint8(mask)*255; % highlight in red
        imwrite(vis, fullfile(outputDirMSER,[baseName '_MSER.png']));

        %% --- Apply CLAHE to segmented image ---
        seg_gray = rgb2gray(vis);
        seg_clahe = CLAHE(seg_gray);   % <-- call your CLAHE.m
        imwrite(seg_clahe, fullfile(outputDirCLAHE,[baseName '_MSER_CLAHE.png']));

        %% --- Metrics ---
        I    = im2uint16(imresize(gray,[512 512]));
        Ith1 = im2uint16(imresize(rgb2gray(vis),[512 512]));
        Ith2 = im2uint16(imresize(seg_clahe,[512 512]));

        metrics1 = [NormalizedAbsoluteError(I,Ith1), NormalizedCrossCorrelation(I,Ith1), ...
                    imageQualityIndex(I,Ith1), HaarPSI(I,Ith1), qilv(I,Ith1,0), ...
                    psnr(I,Ith1), ssim(I,Ith1), FeatureSIM(I,Ith1)];
        metrics2 = [NormalizedAbsoluteError(I,Ith2), NormalizedCrossCorrelation(I,Ith2), ...
                    imageQualityIndex(I,Ith2), HaarPSI(I,Ith2), qilv(I,Ith2,0), ...
                    psnr(I,Ith2), ssim(I,Ith2), FeatureSIM(I,Ith2)];

        if idx==1
            Q_Param = {'NAE','NCC','UIQI','HAAR','QILV','PSNR','SSIM','FSIM'};
            T_mser = table(Q_Param(:), metrics1(:), 'VariableNames', {'Parameter',baseName});
            T_mser_clahe = table(Q_Param(:), metrics2(:), 'VariableNames', {'Parameter',baseName});
        else
            T_mser.(baseName) = metrics1(:);
            T_mser_clahe.(baseName) = metrics2(:);
        end
    end

    % Save metrics separately for each category
    writetable(T_mser, fullfile(outputDirMSER,['results_MSER_' categoryName '.csv']));
    writetable(T_mser_clahe, fullfile(outputDirCLAHE,['results_MSER_CLAHE_' categoryName '.csv']));
end
end

