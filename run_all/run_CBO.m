function run_CBO(bucketPath)
% Process benign and malignant inputs separately
inputDirs = {fullfile(bucketPath,'benign'), fullfile(bucketPath,'malignant')};
outputRoot = bucketPath;

for d = 1:length(inputDirs)
    inputDir = inputDirs{d};
    [~, categoryName] = fileparts(inputDir);

    % Define separate output folders for CBO and CBO+CLAHE
    outputDirCBO   = fullfile(outputRoot, ['output_CBO_' categoryName]);
    outputDirCLAHE = fullfile(outputRoot, ['output_CBO_CLAHE_' categoryName]);
    if ~exist(outputDirCBO,'dir'), mkdir(outputDirCBO); end
    if ~exist(outputDirCLAHE,'dir'), mkdir(outputDirCLAHE); end

    files = dir(fullfile(inputDir,'*.png'));
    files = files(~contains({files.name},{'_CBO','_ACO','_FA','_MSER'})); % avoid recursion

    T_cbo = table; T_cbo_clahe = table;

    for idx = 1:length(files)
        currentImage = fullfile(inputDir, files(idx).name);
        [~, baseName, ~] = fileparts(currentImage);
        img = imread(currentImage);
        gray = im2double(im2gray(img));

        %% --- CBO segmentation ---
        segmentedCBO = CBO_segmentation(gray);
        imwrite(segmentedCBO, fullfile(outputDirCBO,[baseName '_CBO.png']));

        %% --- Apply CLAHE to segmented image ---
        seg_gray = rgb2gray(segmentedCBO);
        seg_clahe = CLAHE(seg_gray);   % <-- call your CLAHE.m
        imwrite(seg_clahe, fullfile(outputDirCLAHE,[baseName '_CBO_CLAHE.png']));

        %% --- Metrics ---
        I    = im2uint16(imresize(gray,[512 512]));
        Ith1 = im2uint16(imresize(rgb2gray(segmentedCBO),[512 512]));
        Ith2 = im2uint16(imresize(seg_clahe,[512 512]));

        metrics1 = [NormalizedAbsoluteError(I,Ith1), NormalizedCrossCorrelation(I,Ith1), ...
                    imageQualityIndex(I,Ith1), HaarPSI(I,Ith1), qilv(I,Ith1,0), ...
                    psnr(I,Ith1), ssim(I,Ith1), FeatureSIM(I,Ith1)];
        metrics2 = [NormalizedAbsoluteError(I,Ith2), NormalizedCrossCorrelation(I,Ith2), ...
                    imageQualityIndex(I,Ith2), HaarPSI(I,Ith2), qilv(I,Ith2,0), ...
                    psnr(I,Ith2), ssim(I,Ith2), FeatureSIM(I,Ith2)];

        if idx==1
            Q_Param = {'NAE','NCC','UIQI','HAAR','QILV','PSNR','SSIM','FSIM'};
            T_cbo = table(Q_Param(:), metrics1(:), 'VariableNames', {'Parameter',baseName});
            T_cbo_clahe = table(Q_Param(:), metrics2(:), 'VariableNames', {'Parameter',baseName});
        else
            T_cbo.(baseName) = metrics1(:);
            T_cbo_clahe.(baseName) = metrics2(:);
        end
    end

    % Save metrics separately for each category
    writetable(T_cbo, fullfile(outputDirCBO,['results_CBO_' categoryName '.csv']));
    writetable(T_cbo_clahe, fullfile(outputDirCLAHE,['results_CBO_CLAHE_' categoryName '.csv']));
end
end

%% --- Inline CBO segmentation with EntropyCost ---
function segmented = CBO_segmentation(gray)
    X = gray(:);
    k = 6; % clusters
    CostFunction = @(m) EntropyCost(m, X);

    MaxIt = 100; nPop = 10; alpha = 0.2;
    VarMin = repmat(min(X),k,1);
    VarMax = repmat(max(X),k,1);

    chefs.Position=[]; chefs.Cost=[]; chefs.Out=[];
    pop=repmat(chefs,nPop,1); BestSol.Cost=inf;

    for i=1:nPop
        pop(i).Position=unifrnd(VarMin,VarMax,[k,1]);
        [pop(i).Cost,pop(i).Out]=CostFunction(pop(i).Position);
        if pop(i).Cost<BestSol.Cost, BestSol=pop(i); end
    end

    for it=1:MaxIt
        newpop=repmat(chefs,nPop,1);
        for i=1:nPop
            newpop(i).Cost=inf;
            for j=1:nPop
                if pop(j).Cost<pop(i).Cost
                    newsol.Position=pop(i).Position+alpha*randn(size(pop(i).Position)).*(pop(j).Position-pop(i).Position);
                    newsol.Position=max(min(newsol.Position,VarMax),VarMin);
                    [newsol.Cost,newsol.Out]=CostFunction(newsol.Position);
                    if newsol.Cost<newpop(i).Cost
                        newpop(i)=newsol;
                        if newpop(i).Cost<BestSol.Cost, BestSol=newpop(i); end
                    end
                end
            end
        end
        pop=[pop;newpop]; [~,so]=sort([pop.Cost]); pop=pop(so); pop=pop(1:nPop);
    end

    CBOlbl=BestSol.Out.ind;
    gray2=reshape(CBOlbl(:,1),size(gray));
    segmented=label2rgb(gray2);
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

