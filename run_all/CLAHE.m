function output = run_CLAHE(image)
    % Ensure grayscale
    if size(image,3) == 3
        image = rgb2gray(image);
    else
        image = im2gray(image);
    end

    % Local contrast enhancement (pre-step)
    edgeThreshold = 0.2;
    amount = 0.5;
    data = localcontrast(image, edgeThreshold, amount);

    % Parameters
    limit = 8;       % window size
    cliplimit = 0.3; % clip limit (0–1)

    % First pass: count pixel occurrences in contextual areas
    eqdata = zeros(size(data,1), size(data,2));
    t = 1; endt = limit;
    for x = 1:size(data,1)
        q = 1; endq = limit;
        for y = 1:size(data,2)
            eqdata(x,y) = 0;
            if (x > t+limit-1), t = t+limit; endt = limit+t-1; end
            if (y > q+limit-1), q = q+limit; endq = limit+q-1; end
            if (endt > size(data,1)), endt = size(data,1); end
            if (endq > size(data,2)), endq = size(data,2); end
            for i = t:endt
                for j = q:endq
                    if data(x,y) == data(i,j)
                        eqdata(x,y) = eqdata(x,y)+1;
                    end
                end
            end
        end
    end

    % Second pass: apply clipping and redistribution
    output = zeros(size(data,1), size(data,2));
    t = 1; endt = limit;
    for x = 1:size(data,1)
        q = 1; endq = limit;
        for y = 1:size(data,2)
            cliptotal = 0; partialrank = 0;
            if (x > t+limit-1), t = t+limit; endt = limit+t-1; end
            if (y > q+limit-1), q = q+limit; endq = limit+q-1; end
            if (endt > size(data,1)), endt = size(data,1); end
            if (endq > size(data,2)), endq = size(data,2); end
            for i = t:endt
                for j = q:endq
                    if eqdata(i,j) > cliplimit
                        incr = cliplimit/eqdata(i,j);
                    else
                        incr = 1;
                    end
                    cliptotal = cliptotal+(1-incr);
                    if data(x,y) > data(i,j)
                        partialrank = partialrank+incr;
                    end
                end
            end
            redistr = (cliptotal/(limit*limit)).*data(x,y);
            output(x,y) = partialrank+redistr;
        end
    end

    % Cast to uint8 for saving/display
    output = uint8(output);
end


