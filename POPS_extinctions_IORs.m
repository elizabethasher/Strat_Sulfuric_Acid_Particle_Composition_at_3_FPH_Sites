
clear
dbs


%% 1.39 IOR
figure
ax = axe;
folderpath = '/Users/abaron/Data/POPS/Boulder_ext1021_IORs/139/';
d = dir(folderpath);
if contains(d(3).name,'NOAA')
    f = d(3:end);
else
    f = d(4:end);
end

t0 = NaT(length(f),1);
fn = string(NaN(length(f),1));
z = 1.8:.1:30;
ext139 = NaN(length(z),length(f));
for i = 1:length(f)

    ik = strfind(f(i).name,'_');
    t0(i) = datetime(f(i).name(1:ik(1)-1),'InputFormat','yyyyMMdd');
    fn(i) = f(i).name(ik(1)+1:end-4);

    file = strcat(folderpath,f(i).name);

    opts = detectImportOptions(file,"FileType","text");
    Ta = readtable(file,opts);
    Ta.Alt_km(Ta.Alt_km<0) = NaN;
    [~,im] = max(Ta.Alt_km);
    Ta = Ta(1:im,:);
    [zo,iz,~] = unique(Ta.Alt_km);
    ext139(:,i) = interp1(zo,Ta.Extinction(iz),z);
    plot(Ta.Extinction, Ta.Alt_km)
end
plot(mean(ext139,2,'omitnan'),z,'LineWidth',5,"Color",'k')

ax.XScale = 'log';
xlabel('Extinction (km^{-1})')
ylabel('Altitude (km)')
ylim([15 25])

%% 1.41 IOR

figure
ax = axe;
folderpath = '/Users/abaron/Data/POPS/Boulder_ext1021_IORs/141/';
d = dir(folderpath);
if contains(d(3).name,'NOAA')
    f = d(3:end);
else
    f = d(4:end);
end

t0 = NaT(length(f),1);
fn = string(NaN(length(f),1));
z = 1.8:.1:30;
ext141 = NaN(length(z),length(f));
for i = 1:length(f)

    ik = strfind(f(i).name,'_');
    t0(i) = datetime(f(i).name(1:ik(1)-1),'InputFormat','yyyyMMdd');
    fn(i) = f(i).name(ik(1)+1:end-4);

    file = strcat(folderpath,f(i).name);

    opts = detectImportOptions(file,"FileType","text");
    Ta = readtable(file,opts);
    Ta.Alt_km(Ta.Alt_km<0) = NaN;
    [~,im] = max(Ta.Alt_km);
    Ta = Ta(1:im,:);
    [zo,iz,~] = unique(Ta.Alt_km);
    ext141(:,i) = interp1(zo,Ta.Extinction(iz),z);
    plot(Ta.Extinction, Ta.Alt_km)
end
plot(mean(ext141,2,'omitnan'),z,'LineWidth',5,"Color",'k')

ax.XScale = 'log';
xlabel('Extinction (km^{-1})')
ylabel('Altitude (km)')
ylim([15 25])

%% 1.43 IOR
figure
ax = axe;
folderpath = '/Users/abaron/Data/POPS/Boulder_ext1021_IORs/143/';
d = dir(folderpath);
if contains(d(3).name,'NOAA')
    f = d(3:end);
else
    f = d(4:end);
end

t0 = NaT(length(f),1);
fn = string(NaN(length(f),1));
z = 1.8:.1:30;
ext143 = NaN(length(z),length(f));
for i = 1:length(f)

    ik = strfind(f(i).name,'_');
    t0(i) = datetime(f(i).name(1:ik(1)-1),'InputFormat','yyyyMMdd');
    fn(i) = f(i).name(ik(1)+1:end-4);

    file = strcat(folderpath,f(i).name);

    opts = detectImportOptions(file,"FileType","text");
    Ta = readtable(file,opts);
    Ta.Alt_km(Ta.Alt_km<0) = NaN;
    [~,im] = max(Ta.Alt_km);
    Ta = Ta(1:im,:);
    [zo,iz,~] = unique(Ta.Alt_km);
    ext143(:,i) = interp1(zo,Ta.Extinction(iz),z);
    plot(Ta.Extinction, Ta.Alt_km)
end
plot(mean(ext143,2,'omitnan'),z,'LineWidth',5,"Color",'k')

ax.XScale = 'log';
xlabel('Extinction (km^{-1})')
ylabel('Altitude (km)')
ylim([15 25])


%% Comparison and differences

m139 = lisbi23(mean(ext139,2,'omitnan'),1,3);
m141 = lisbi23(mean(ext141,2,'omitnan'),1,3);
m143 = lisbi23(mean(ext143,2,'omitnan'),1,3);

figure
ax = axe;
plot(m139,z,'LineWidth',3,'DisplayName','H_2SO_4 Wt = 0.55')
plot(m141,z,'LineWidth',3,'DisplayName','H_2SO_4 Wt = 0.75')
plot(m143,z,'LineWidth',3,'DisplayName','H_2SO_4 Wt = 0.95')
ax.XScale = 'log';
xlabel('Extinction (km^{-1})')
ylabel('Altitude (km)')
ylim([15 25])
xlim([1e-5 1e-3])
legend

m139 = lisbi23(mean(ext139,2,'omitnan'),1,3);
m141 = lisbi23(mean(ext141,2,'omitnan'),1,3);
m143 = lisbi23(mean(ext143,2,'omitnan'),1,3);

d1 = abs(100.*(m139-m141)./m141);
d2 = abs(100.*(m143-m141)./m143);

figure
axe;
plot(d1,z,'LineWidth',3,'DisplayName','H_2SO_4 Wt 0.55 - 0.75')
plot(d2,z,'LineWidth',3,'DisplayName','H_2SO_4 Wt 0.95 - 0.75')
xlabel('\DeltaExtinction (%)')
ylabel('Altitude (km)')
ylim([15 25])
xlim([0 10])
legend

x = .35:.001:.99;
y = 1.33 + .105*x;

figure
axe;
plot(x,y,'linewidth',3,'displayname','IOR = 1.33 + 0.105 Wt')
xlabel('H_2SO_4 Weight Fraction (Wt)')
ylabel('IOR @ 1020 nm - Real part')
xlim([.35 1])
legend