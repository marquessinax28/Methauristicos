function run_FA(bucketPath)

% Process benign and malignant inputs separately
inputDirs = {fullfile(bucketPath,'benign'), fullfile(bucketPath,'malignant')};
outputRoot = bucketPath;

for d = 1:length(inputDirs)
    inputDir = inputDirs{d};
    [~, categoryName] = fileparts(inputDir);

    % Define separate output folders for FA and FA+CLAHE
    outputDirFA = fullfile(outputRoot, ['output_FA_' categoryName]);
    outputDirCLAHE = fullfile(outputRoot, ['output_FA_CLAHE_' categoryName]);
    if ~exist(outputDirFA,'dir'), mkdir(outputDirFA); end
    if ~exist(outputDirCLAHE,'dir'), mkdir(outputDirCLAHE); end

    files = dir(fullfile(inputDir,'*.png'));
    files = files(~contains({files.name},{'_MSER','_FA'})); % avoid recursion

    T_fa = table; T_fa_clahe = table;

    for idx = 1:length(files)
        currentImage = fullfile(inputDir, files(idx).name);
        [~, baseName, ~] = fileparts(currentImage);
        img = imread(currentImage);

        % Cast to double for math, keep grayscale
        gray = im2gray(img);
        gray_dbl = double(gray);

        %% --- FA clustering with EntropyCost ---
        X = gray_dbl(:);   % double vector
        k = 6;
        CostFunction = @(m) EntropyCost(m,X);
        VarSize = [k size(X,2)];
        VarMin = repmat(min(X),k,1);
        VarMax = repmat(max(X),k,1);

        MaxIt=20; nPop=10; gamma=1; beta0=2; alpha=0.2; alpha_damp=0.98;
        delta=0.05*(VarMax-VarMin); m=2; dmax=norm(VarMax-VarMin);

        firefly.Position=[]; firefly.Cost=[]; firefly.Out=[];
        pop=repmat(firefly,nPop,1); BestSol.Cost=inf;

        for i=1:nPop
            pop(i).Position=unifrnd(VarMin,VarMax,VarSize);
            [pop(i).Cost,pop(i).Out]=CostFunction(pop(i).Position);
            if pop(i).Cost<=BestSol.Cost, BestSol=pop(i); end
        end

        for it=1:MaxIt
            newpop=repmat(firefly,nPop,1);
            for i=1:nPop
                newpop(i).Cost=inf;
                for j=1:nPop
                    if pop(j).Cost<pop(i).Cost
                        rij=norm(pop(i).Position-pop(j).Position)/dmax;
                        beta=beta0.*exp(-gamma.*rij^m);
                        e=delta.*unifrnd(-1,+1,VarSize);
                        newsol.Position=pop(i).Position+beta.*rand(VarSize).*(pop(j).Position-pop(i).Position)+alpha.*e;
                        newsol.Position=max(newsol.Position,VarMin);
                        newsol.Position=min(newsol.Position,VarMax);
                        [newsol.Cost,newsol.Out]=CostFunction(newsol.Position);
                        if newsol.Cost<=newpop(i).Cost
                            newpop(i)=newsol;
                            if newpop(i).Cost<=BestSol.Cost, BestSol=newpop(i); end
                        end
                    end
                end
            end
            pop=[pop;newpop]; [~,so]=sort([pop.Cost]); pop=pop(so); pop=pop(1:nPop);
            alpha=alpha*alpha_damp;
        end

        % Segmented image (FA only)
        FAlbl=BestSol.Out.ind;
        seg_fa=label2rgb(reshape(FAlbl(:,1),size(gray)));
        imwrite(seg_fa, fullfile(outputDirFA,[baseName '_FA.png']));

        %% --- Apply CLAHE to segmented image ---
        seg_fa_gray = rgb2gray(seg_fa);
        seg_fa_clahe = CLAHE(seg_fa_gray);  % <-- call your CLAHE.m
        imwrite(seg_fa_clahe, fullfile(outputDirCLAHE,[baseName '_FA_CLAHE.png']));

        %% --- Metrics ---
        I=im2uint16(imresize(gray,[512 512]));
        Ith1=im2uint16(imresize(rgb2gray(seg_fa),[512 512]));
        Ith2=im2uint16(imresize(seg_fa_clahe,[512 512]));

        metrics1=[NormalizedAbsoluteError(I,Ith1),NormalizedCrossCorrelation(I,Ith1), ...
                  imageQualityIndex(I,Ith1),HaarPSI(I,Ith1),qilv(I,Ith1,0), ...
                  psnr(I,Ith1),ssim(I,Ith1),FeatureSIM(I,Ith1)];
        metrics2=[NormalizedAbsoluteError(I,Ith2),NormalizedCrossCorrelation(I,Ith2), ...
                  imageQualityIndex(I,Ith2),HaarPSI(I,Ith2),qilv(I,Ith2,0), ...
                  psnr(I,Ith2),ssim(I,Ith2),FeatureSIM(I,Ith2)];

        if idx==1
            Q_Param={'NAE','NCC','UIQI','HAAR','QILV','PSNR','SSIM','FSIM'};
            T_fa=table(Q_Param(:),metrics1(:),'VariableNames',{'Parameter',baseName});
            T_fa_clahe=table(Q_Param(:),metrics2(:),'VariableNames',{'Parameter',baseName});
        else
            T_fa.(baseName)=metrics1(:);
            T_fa_clahe.(baseName)=metrics2(:);
        end
    end

    % Save metrics separately for each category
    writetable(T_fa, fullfile(outputDirFA,['results_FA_' categoryName '.csv']));
    writetable(T_fa_clahe, fullfile(outputDirCLAHE,['results_FA_CLAHE_' categoryName '.csv']));
end
end

function [cost, out] = EntropyCost(m, X)
    [~, ind] = min(abs(X - m'), [], 2);
    p = histcounts(ind, 1:length(m)+1);
    p = p / sum(p);
    entropyValue = -sum(p(p > 0) .* log2(p(p > 0)));
    cost = -entropyValue;
    out.ind = ind;
end

