% This Matlab script plot a raster plot of the output neural activity
load 'output.dat';
lx=max(output(:,1));

spk=sort(output(:,2)');

nspk=[];
if length(spk) > 0
    reps=1;
    nspk(1)=spk(1);
else
    reps=0;
end
for n=2:length(spk),
    if spk(n-1) ~= spk(n)
        reps=reps+1;
        nspk(reps)=spk(n);
    end
end

tot_spks=0;
for n=1:reps
    tspk=output(find(output(:,2)==nspk(n)),1);
    tot_spks=tot_spks+length(tspk);
    line((tspk*ones(1,2))',(ones(length(tspk),1)*[n-0.25,n+0.25])','Color','b');
end
axis tight
xlabel('time');
ylabel('neuron number');
display(['Total number of spikes: ' num2str(tot_spks)]);
display(['Number of spiking neurons: ' num2str(reps)]);
set(gca,'YTick',1:reps);
set(gca,'YTickLabel',nspk);
