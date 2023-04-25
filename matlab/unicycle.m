function out = unicycle(obs_loc, goal, alpha)

    %% Control barrier functions (CBF) for unicycle model
    % close all
    
%     obs_loc = cell2mat(obs_loc);
%     goal = cell2mat(goal);

%     phi = [3 -0.0146];
    phi = [1.0949078  -0.30310962];
    
    %% Problem setting
    % Parameters of problem
%     par.xgoal=[4;0];                   % goal point
    par.xgoal=goal;
%     par.xO=[[x_obstacle,y_obstacle]];          % obstacle center
    par.xO=obs_loc;
    par.DO=0.5*ones(1,size(par.xO,2));	% obstacle radius	% 0.75, 0.82
    
    % Controller parameters
    par.Kv=0.08;	% gain of desired controller	% 0.08, 0.16
    par.Kom=0.4;	% gain of desired controller	%  0.4, 0.8
%     par.alpha=phi(1);	% CBF parameter
    par.alpha=alpha;
    par.delta=0.5;  % CBF parameter
    par.R=0.5;      % scaling parameter for cost, (v-vd)^2+R^2*(om-omd)^2	% 0.5, 0.25
    
    % Initial conditions
    xinit=[0;0;0];
    
    % Dimension of problem
    par.dim=size(par.xgoal,1);
    
    % Simulation settings
    t0=0;
    tend=60;
    dt=0.01;
    tsim=t0:dt:tend;
    ode_opts=odeset('RelTol',1e-6);
    
    % Plot settings
    tminplot=t0; tmaxplot=tend;
    xminplot=0; xmaxplot=4;
    yminplot=-3; ymaxplot=1;
    hminplot=-0.2; hmaxplot=0.8;
    hdotminplot=-10; hdotmaxplot=10;
    vminplot=0; vmaxplot=0.5;
    uminplot=0; umaxplot=0.5;
    purple=[170,0,170]/256;
    orange=[255,170,0]/256;
    black=[0,0,0];
    gray=[0.5,0.5,0.5];
    darkgreen=[0,170,0]/256;
    darkred=[230,0,0]/256;
    
    %% Simulation
    % Simulation of CBF design with unicycle
    [tU,xU]=ode45(@(t,x)ClosedLoop(t,x,@RHS_unicycle,@K_CBF,par),tsim,xinit,ode_opts);
    hU=CBF(xU.',par).';
    [vsafeU,vdesU]=K_CBF(xU.',par); vsafeU =vsafeU.'; vdesU=vdesU.';
    
%     %% Plot results
%     % Plot of trajectory
%     figure(1); clf;
%     subplot(2,2,1); hold on; grid on; box on;
%     PlotPoints(xinit,par.xgoal,darkgreen,darkred);
%     PlotObstacles(par.xO,par.DO);
%     plot(xU(:,1),xU(:,2),'Color',orange,'LineWidth',2,'DisplayName','unicycle');
%     set(gca,'ColorOrderIndex',1);
%     PlotFinalize([],{'position, $q_1$ (m)','position, $q_2$ (m)','time, $t$ (s)'},...
%                 [xminplot,xmaxplot,yminplot,ymaxplot,tminplot,tmaxplot],...
%                 [(xmaxplot-xminplot)/(ymaxplot-yminplot),1,1]);
    
%     % Plot of velocity
%     subplot(2,2,2); hold on; box on;
%     plot(tU,vdesU(:,1),'Color',darkgreen,'LineWidth',2,'HandleVisibility','off');
%     plot(tU,vsafeU(:,1),'Color',darkred,'LineWidth',2,'HandleVisibility','off');
%     % plot(tU,vdesU(:,2),'--','Color',darkgreen,'LineWidth',2,'HandleVisibility','off');
%     % plot(tU,vsafeU(:,2),'--','Color',darkred,'LineWidth',2,'HandleVisibility','off');
%     set(gca,'ColorOrderIndex',1);
%     PlotFinalize([],{'time, $t$ (s)','forward velocity, $v$ (m/s)'},...
%                 [tminplot,tmaxplot,vminplot,vmaxplot],[1,1,1]);
%     
%     % Plot of safety
%     subplot(2,2,3); hold on; box on;
%     plot([tminplot,tmaxplot],[0,0],'k','LineWidth',1);
%     plot(tU,hU,'Color',orange,'LineWidth',2,'HandleVisibility','off');
%     set(gca,'ColorOrderIndex',1);
%     PlotFinalize([],{'time, $t$ (s)','CBF, $h$ (m)'},...
%                 [tminplot,tmaxplot,hminplot,hmaxplot],[1,1,1]);
    
    % % Plot of control input
    % subplot(2,2,4); hold on; box on;
    % % plot(tD,vecnorm(uD.'),'Color',purple,'LineWidth',2,'HandleVisibility','off');
    % set(gca,'ColorOrderIndex',1);
    % PlotFinalize([],{'time, $t$ (s)','input, $u$ (m/s$^2$)'},...
    %             [tminplot,tmaxplot,uminplot,umaxplot],[1,1,1]);
    
    %% Functions defining the dynamics
    % Closed loop dynamics
    function dzdt = ClosedLoop(~,z,RHS,K,par)
        u = K(z,par);
        dzdt = RHS(z,u,par);
    end
    
    % Right hand side of unicycle plant
    function dzdt = RHS_unicycle(z,u,~)
        v = u(1);
        om = u(2);
        psi = z(3);
        dzdt = [v*cos(psi);
                v*sin(psi);
                om];
    end
    
    % Desired controller
    function vdes = K_des(z,par)
        xgoal = par.xgoal;
        dim = par.dim;
        Kv = par.Kv;    
        Kom = par.Kom;    
        x = z(1:dim,:);
        psi = z(dim+1,:);
        vdes = [Kv*vecnorm(x-xgoal);...
                -Kom*(sin(psi)-(xgoal(2)-x(2,:))./vecnorm(x-xgoal))];
    end
    
    % CBF evaluation
    function [h,Lfh,Lgh,LghLgh] = CBF(z,par)
        dim = par.dim;
        xO = par.xO;
        DO = par.DO;
        delta = par.delta;
        x = z(1:dim,:);
        psi = z(dim+1,:);
        tpsi = [cos(psi);sin(psi)];
        npsi = [-sin(psi);cos(psi)];
        
        % control barrier function
        hk = nan(length(DO),size(x,2));
        for kobs=1:length(DO)
            xobs = xO(:,kobs);
            robs = DO(kobs);
            dobs = vecnorm(x-double(xobs));
            
    %         hk(kobs,:) = dobs-robs;
    
            nobs = (x-double(xobs))./dobs;
            nobstpsi=sum(nobs.*tpsi,1);
            hk(kobs,:) = dobs-robs+delta*nobstpsi;
        end
        
        % use closest obstacle only
        [~,idx] = min(hk,[],1);
        d = vecnorm(x-double(xO(:,idx)));
        r = DO(idx);
        nO = (x-double(xO(:,idx)))./d;
        nOtpsi=sum(nO.*tpsi,1);
        Lfh = 0;        % f=0
        
    %     h = d-r;
    %     Lgh = [eOepsi;zeros(size(eOepsi))].';
    %     LghLgh = sum(Lgh.*Lgh,2).';
        
        h = d-r+delta*nOtpsi;
        nOnpsi=sum(nO.*npsi,1);
        Lgh = [nOtpsi+delta./d.*(1-nOtpsi.^2); delta*nOnpsi].';
        LghLgh = sum(Lgh.*Lgh,2).';
    end
    
    % CBF controller
    function [vsafe,vdes] = K_CBF(z,par)
        alpha = par.alpha;
        R = par.R;
        % desired control input
        vdes = K_des(z,par);
        % safety filter
        [h,Lfh,Lgh,LghLgh] = CBF(z,par);
        W=[1,0;0,1/R];
        phi = Lfh + sum(Lgh.*vdes.',2).' + alpha*h;	% or diag(Lgh*udes).'
    %     vsafe = vdes + max(0,-phi).*Lgh.'./LghLgh;	% analytical solution of QP
        vsafe = vdes + W*(max(0,-phi).*(Lgh*W).')./sum((Lgh*W).*(Lgh*W),2).';	% analytical solution of QP
    end
    
    %% Functions for plotting
    % Plot the start, goal and intermediate points
    function PlotPoints(xinit,xgoal,color1,color2)
        plot(xinit(1),xinit(2),'.','Color',color1,'Markersize',20,'DisplayName','Start');
        plot(xgoal(1),xgoal(2),'.','Color',color2,'Markersize',20,'DisplayName','Goal');
    end
    
    % Plot the obstacles to be avoided
    function PlotObstacles(xO,DO)
        phi=0:2*pi/100:2*pi;
        xcircle=[cos(phi);sin(phi)];
        plot(xO(1,1),xO(2,1),'k.','Markersize',20,'HandleVisibility','off');
        plot(xO(1,1)+DO(1)*xcircle(1,:),xO(2,1)+DO(1)*xcircle(2,:),...
            'k','LineWidth',2,'DisplayName','Avoid');
        for kobs=2:size(xO,2)
            plot(xO(1,kobs),xO(2,kobs),'k.','Markersize',20,'HandleVisibility','off');
            plot(xO(1,kobs)+DO(kobs)*xcircle(1,:),xO(2,kobs)+DO(kobs)*xcircle(2,:),...
                'k','LineWidth',2,'HandleVisibility','off');
        end
    end
    
    % Finalize plot with axis limits, labels, legends, title
    function PlotFinalize(titletext,axislabels,axislimits,aspectratio)
        axis(axislimits);
        pbaspect(aspectratio);
        xlabel(axislabels{1},'Interpreter','latex','FontSize',18);
        ylabel(axislabels{2},'Interpreter','latex','FontSize',18);
        if length(axislabels)>2
            zlabel(axislabels{3},'Interpreter','latex','FontSize',18);
        end
        set(gca,'TickLabelInterpreter','latex','FontSize',12);
    %     legend('Location','SW','Interpreter','latex','FontSize',14);
        if isempty(get(get(gca,'Legend'),'String'))
            legend off;
        end
        title(titletext,'Interpreter','latex','FontSize',16);
    end

    out.tU = tU;
    out.xU = xU;
    out.hU = hU;

end