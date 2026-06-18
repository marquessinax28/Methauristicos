function run_ACO(bucketPath)
% Process benign and malignant inputs separately
inputDirs = {fullfile(bucketPath,'benign'), fullfile(bucketPath,'malignant')};
outputRoot = bucketPath;

for d = 1:length(inputDirs)
    inputDir = inputDirs{d};
    [~, categoryName] = fileparts(inputDir);

    % Define separate output folders for ACO and ACO+CLAHE
    outputDirACO   = fullfile(outputRoot, ['output_ACO_' categoryName]);
    outputDirCLAHE = fullfile(outputRoot, ['output_ACO_CLAHE_' categoryName]);
    if ~exist(outputDirACO,'dir'), mkdir(outputDirACO); end
    if ~exist(outputDirCLAHE,'dir'), mkdir(outputDirCLAHE); end

    files = dir(fullfile(inputDir,'*.png'));
    files = files(~contains({files.name},{'_ACO','_FA','_MSER'})); % avoid recursion

    T_aco = table; T_aco_clahe = table;

    for idx = 1:length(files)
        currentImage = fullfile(inputDir, files(idx).name);
        [~, baseName, ~] = fileparts(currentImage);
        img = imread(currentImage);
        gray = im2double(im2gray(img));

        %% --- ACO segmentation ---
        segmentedACO = ACO_segmentation(gray);
        imwrite(segmentedACO, fullfile(outputDirACO,[baseName '_ACO.png']));

        %% --- Apply CLAHE to segmented image ---
        seg_gray = rgb2gray(segmentedACO);
        seg_clahe = CLAHE(seg_gray);   % <-- call your CLAHE.m
        imwrite(seg_clahe, fullfile(outputDirCLAHE,[baseName '_ACO_CLAHE.png']));

        %% --- Metrics ---
        I    = im2uint16(imresize(gray,[512 512]));
        Ith1 = im2uint16(imresize(rgb2gray(segmentedACO),[512 512]));
        Ith2 = im2uint16(imresize(seg_clahe,[512 512]));

        metrics1 = [NormalizedAbsoluteError(I,Ith1), NormalizedCrossCorrelation(I,Ith1), ...
                    imageQualityIndex(I,Ith1), HaarPSI(I,Ith1), qilv(I,Ith1,0), ...
                    psnr(I,Ith1), ssim(I,Ith1), FeatureSIM(I,Ith1)];
        metrics2 = [NormalizedAbsoluteError(I,Ith2), NormalizedCrossCorrelation(I,Ith2), ...
                    imageQualityIndex(I,Ith2), HaarPSI(I,Ith2), qilv(I,Ith2,0), ...
                    psnr(I,Ith2), ssim(I,Ith2), FeatureSIM(I,Ith2)];

        if idx==1
            Q_Param = {'NAE','NCC','UIQI','HAAR','QILV','PSNR','SSIM','FSIM'};
            T_aco = table(Q_Param(:), metrics1(:), 'VariableNames', {'Parameter',baseName});
            T_aco_clahe = table(Q_Param(:), metrics2(:), 'VariableNames', {'Parameter',baseName});
        else
            T_aco.(baseName) = metrics1(:);
            T_aco_clahe.(baseName) = metrics2(:);
        end
    end

    % Save metrics separately for each category
    writetable(T_aco, fullfile(outputDirACO,['results_ACO_' categoryName '.csv']));
    writetable(T_aco_clahe, fullfile(outputDirCLAHE,['results_ACO_CLAHE_' categoryName '.csv']));
end
end

%% --- Inline ACO segmentation with EntropyCost ---
function segmented = ACO_segmentation(gray)
    X = gray(:);
    k = 6; % clusters
    CostFunction = @(m) EntropyCost(m, X);

    VarSize = [k size(X,2)];
    VarMin = repmat(min(X),k,1);
    VarMax = repmat(max(X),k,1);

    MaxIt = 100; nAnt = 20; ArchiveSize = 50;
    q = 0.7; xi = 0.3; xi_damp = 0.98;

    ant.Position = []; ant.Cost = []; ant.Out = [];
    Archive = repmat(ant, ArchiveSize, 1);
    for i = 1:ArchiveSize
        Archive(i).Position = unifrnd(VarMin, VarMax, VarSize);
        [Archive(i).Cost, Archive(i).Out] = CostFunction(Archive(i).Position);
    end
    [~, SortOrder] = sort([Archive.Cost]); Archive = Archive(SortOrder);
    BestSol = Archive(1);

    for it = 1:MaxIt
        w = 1/(sqrt(2*pi)*q*ArchiveSize) * exp(-0.5*((1:ArchiveSize)/(q*ArchiveSize)).^2);
        w = w/sum(w);
        newArchive = repmat(ant, nAnt, 1);
        for i = 1:nAnt
            R = rand; S = 0;
            for j = 1:ArchiveSize
                S = S + w(j);
                if S >= R, Elite = Archive(j); break; end
            end
            sigma = xi*abs(Elite.Position - VarMin);
            newAnt.Position = Elite.Position + sigma.*randn(size(Elite.Position));
            newAnt.Position = max(newAnt.Position, VarMin);
            newAnt.Position = min(newAnt.Position, VarMax);
            [newAnt.Cost, newAnt.Out] = CostFunction(newAnt.Position);
            newArchive(i) = newAnt;
        end
        Archive = [Archive; newArchive];
        [~, SortOrder] = sort([Archive.Cost]); Archive = Archive(SortOrder);
        Archive = Archive(1:ArchiveSize);
        if Archive(1).Cost < BestSol.Cost, BestSol = Archive(1); end
        xi = xi * xi_damp;
    end

    ACOlbl = BestSol.Out.ind;
    gray2 = reshape(ACOlbl(:,1), size(gray));
    segmented = label2rgb(gray2);
end

%% --- Inline EntropyCost function ---
function [cost, out] = EntropyCost(m, X)
    [~, ind] = min(abs(X - m'), [], 2);
    p = histcounts(ind, 1:length(m)+1);
    p = p / sum(p);
    entropyValue = -sum(p(p > 0) .* log2(p(p > 0)));
    cost = -entropyValue;
    out.ind = ind;
end

