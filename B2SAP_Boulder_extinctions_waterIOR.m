

clear
dbs

lsage = 1021;

folderpath = '/Users/abaron/Data/POPS/ICARTT_files/ESSD_Paper_2026/NOAA-POPS-B2SAP_GMLBalloon_Boulder_100m_v1/';

d = dir(folderpath);

if contains(d(3).name,'NOAA')
    f = d(3:end);
else
    f = d(4:end);
end


t0 = NaT(length(f),1);
fn = string(NaN(length(f),1));

for i = 1:length(f)

    ik = strfind(f(i).name,'_');
    t0(i) = datetime(f(i).name(ik(3)+1:ik(4)-1),'InputFormat','yyyyMMdd');
    fn(i) = f(i).name(ik(5)+1:ik(6)-1);

    file = [folderpath f(i).name];

    opts = detectImportOptions(file,"FileType","text");
    Ta = readtable(file,opts);

    fid = fopen(file, 'r');
    ih =0;
    while ih ==0
        rawHeader = fgetl(fid);
        if contains(rawHeader,'OTHER_COMMENTS')
            ih=1;
        end
    end
    fclose(fid);

    dpe = str2double(regexp(rawHeader, '\d+\.?\d*', 'match'));
    dpm = (dpe(1:end-1)+dpe(2:end))./2;
    dlogdp = diff(dpe)./dpm./log(10);
    bins = table2array(Ta(:,16:end));
    dNdlogDp = bins./dlogdp;

    %% Mie calculation
    ior = 1.43; % Complex index of refraction
    wv = 1021; % wavelength of computation
    dens = 1.76; % particle density g cm-3

    n = height(Ta);
    mc = NaN(n,6);

    for ie = 1:n
        sd = dNdlogDp(ie,:);
        mc(ie,:) = SizeDist_Optics(ior, dpm, sd, wv, 'density',dens, 'vectorout',true,'normalized',false,'nobackscat',true);
    end

    Ts = Ta(:,1:12);
    Ts.Extinction = mc(:,1)./1000; % km-1

    sfilename = strcat('/Users/abaron/Data/POPS/Boulder_ext1021_IORs/143/',...
        f(i).name(ik(3)+1:ik(4)-1),'_',fn(i),'.csv');
    writetable(Ts,sfilename)
end
