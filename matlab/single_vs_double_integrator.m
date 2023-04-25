%% Control barrier functions (CBF) for single integrator model
function out = single_vs_double_integrator(obs_loc, goal, alpha)

    % close all
    
    phi = [1.0949078  -0.30310962];
    
    %% Problem setting
    % Parameters of problem
%     par.xgoal=[4;0];                   % goal point
    par.xgoal=goal;
    % par.xO=[[1.5;0],[3;-1.5]];          % obstacle center
%     par.xO=[[x_obstacle;y_obstacle]];          % obstacle center
    par.xO=obs_loc;
    par.DO=0.5*ones(1,size(par.xO,2));	% obstacle radius
    
    % Controller parameters
    par.Kp=0.1;     % gain of desired controller
%     par.alpha=phi(1);	% CBF parameter	% 0.1, 0.2, 0.5, 1
    par.alpha=alpha;
    par.Kv=0.5;       % gain of tracking controller
    
    % % Parameters of quadruped experiment -- quadruped dynamics, not double integrator
    % r_a1=0.2794;
    % par.xgoal=[4;-1];                           % goal point
    % par.xO=[[1.5;0],[3;-1.5]];                  % obstacle center
    % par.DO=(0.5+r_a1)*ones(1,size(par.xO,2));	% obstacle radius
    % scale=0.05;
    % par.Kp=1*scale;
    % par.alpha=0.2;
    % par.Kv=5*scale;
    
    % Initial conditions
    xinit=[0;0];
    vinit=[0;0];
    yinit=[xinit;vinit];
    
    % Dimension of problem
    par.dim=size(par.xgoal,1);
    
    % Simulation settings
    t0=0;
    tend=45;
    dt=0.01;
    tsim=t0:dt:tend;
    ode_opts=odeset('RelTol',1e-6);
    
    % Plot settings
    tminplot=t0; tmaxplot=tend;
    xminplot=0; xmaxplot=4;
    yminplot=-2; ymaxplot=2;
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
    % Simulation of CBF design with single integrator
    [tS,xS]=ode45(@(t,x)ClosedLoop(t,x,@RHS_single,@K_CBF,par),tsim,xinit,ode_opts);
    hS=CBF(xS.',par).';
    [vsafeS,vdesS]=K_CBF(xS.',par); vsafeS =vsafeS.'; vdesS=vdesS.';
    
    % Simulation of CBF design with double integrator
    [tD,yD]=ode45(@(t,y)ClosedLoop(t,y,@RHS_double,@K_CBF,par),tsim,yinit,ode_opts);
    xD=yD(:,1:par.dim); vD=yD(:,par.dim+1:end);
    hD=CBF(yD.',par).';
    [vsafeD,vdesD]=K_CBF(yD.',par); vsafeD =vsafeD.'; vdesD=vdesD.';
    uD=-par.Kv*(vD-vsafeD);
    
%     %% Plot results
%     % Plot of trajectory
%     figure(1); clf;
%     subplot(1,2,1); hold on; grid on; box on;
%     PlotPoints(xinit,par.xgoal,darkgreen,darkred);
%     PlotObstacles(par.xO,par.DO);
%     plot(xS(:,1),xS(:,2),'Color',black,'LineWidth',2,'DisplayName','single integrator');
%     plot(xD(:,1),xD(:,2),'Color',purple,'LineWidth',2,'DisplayName','double integrator');
%     set(gca,'ColorOrderIndex',1);
%     PlotFinalize([],{'position, $q_1$ (m)','position, $q_2$ (m)','time, $t$ (s)'},...
%                 [xminplot,xmaxplot,yminplot,ymaxplot,tminplot,tmaxplot],...
%                 [(xmaxplot-xminplot)/(ymaxplot-yminplot),1,1]);
    
    % % Plot of velocity
    % subplot(2,2,2); hold on; box on;
    % % plot([tminplot,tmaxplot],[0,0],'k','LineWidth',1);
    % % plot(tS,vecnorm(vdesS.'),'Color',black,'LineWidth',2,'DisplayName','double integrator');
    % % plot(tS,vecnorm(vsafeS.'),'Color',black,'LineWidth',2,'DisplayName','double integrator');
    % % plot(tD,vdesD(:,1),'Color',darkgreen,'LineWidth',2,'HandleVisibility','off');
    % % plot(tD,vdesD(:,2),'--','Color',darkgreen,'LineWidth',2,'HandleVisibility','off');
    % % plot(tD,vsafeD(:,1),'Color',darkred,'LineWidth',2,'HandleVisibility','off');
    % % plot(tD,vsafeD(:,2),'--','Color',darkred,'LineWidth',2,'HandleVisibility','off');
    % % plot(tD,vD(:,1),'Color',purple,'LineWidth',2,'HandleVisibility','off');
    % % plot(tD,vD(:,2),'--','Color',purple,'LineWidth',2,'HandleVisibility','off');
    % plot(tD,vecnorm(vdesD.'),'Color',darkgreen,'LineWidth',2,'HandleVisibility','off');
    % plot(tD,vecnorm(vsafeD.'),'Color',darkred,'LineWidth',2,'HandleVisibility','off');
    % plot(tD,vecnorm(vD.'),'Color',purple,'LineWidth',2,'HandleVisibility','off');
    % set(gca,'ColorOrderIndex',1);
    % PlotFinalize([],{'time, $t$ (s)','velocity, $\|\dot{q}\|$ (m/s)'},...
    %             [tminplot,tmaxplot,vminplot,vmaxplot],[1,1,1]);
    
%     % Plot of safety
%     subplot(1,2,2); hold on; box on;
%     plot([tminplot,tmaxplot],[0,0],'k','LineWidth',1);
%     plot(tS,hS,'Color',black,'LineWidth',2,'HandleVisibility','off');
%     plot(tD,hD,'Color',purple,'LineWidth',2,'HandleVisibility','off');
%     set(gca,'ColorOrderIndex',1);
%     PlotFinalize([],{'time, $t$ (s)','CBF, $h$ (m)'},...
%                 [tminplot,tmaxplot,hminplot,hmaxplot],[1,1,1]);
    
    % % Plot of control input
    % subplot(2,2,4); hold on; box on;
    % % plot([tminplot,tmaxplot],[0,0],'k','LineWidth',1);
    % % plot(tD,uD(:,1),'Color',purple,'LineWidth',2,'HandleVisibility','off');
    % % plot(tD,uD(:,2),'--','Color',purple,'LineWidth',2,'HandleVisibility','off');
    % plot(tD,vecnorm(uD.'),'Color',purple,'LineWidth',2,'HandleVisibility','off');
    % set(gca,'ColorOrderIndex',1);
    % PlotFinalize([],{'time, $t$ (s)','input, $u$ (m/s$^2$)'},...
    %             [tminplot,tmaxplot,uminplot,umaxplot],[1,1,1]);
    
    %% Functions defining the dynamics
    % Closed loop dynamics
    function dzdt = ClosedLoop(~,z,RHS,K,par)
        u = K(z,par);
        dzdt = RHS(z,u,par);
    end
    
    % Right hand side of single integrator plant
    function dzdt = RHS_single(~,u,~)
        dzdt = u;
    end
    
    % Right hand side of double integrator plant
    function dzdt = RHS_double(z,u,par)
        dim=par.dim;
        Kv = par.Kv;
        v = z(dim+1:end);
        dzdt = [v;-Kv*(v-u)];
    end
    
    % Desired controller
    function vdes = K_des(z,par)
        xgoal = par.xgoal;
        dim = par.dim;
        Kp = par.Kp;    
        x = z(1:dim,:);
        vdes = -Kp*(x-xgoal);
    end
    
    % CBF evaluation
    function [h,Lfh,Lgh,LghLgh] = CBF(z,par)
        dim = par.dim;
        xO = par.xO;
        DO = par.DO;
        x = z(1:dim,:);
        % control barrier function
        hk = nan(length(DO),size(x,2));
        for kobs=1:length(DO)
            xobs = xO(:,kobs);
            Dobs = DO(kobs);
%             disp(class(x));disp(x);disp(class(xobs));disp(xobs);disp(class(Dobs));disp(Dobs);disp(class(kobs));disp(kobs)
            hk(kobs,:) = vecnorm(x-double(xobs))-Dobs;
        end
        % use closest obstacle only
        [h,idx] = min(hk,[],1);
%         disp(class(x));disp(x);disp(class(xO(:,idx)));disp(xO(:,idx))
        gradh = ((x-double(xO(:,idx)))./vecnorm(x-double(xO(:,idx)))).';
        Lfh = 0;        % f=0
        Lgh = gradh;    % g=I
        LghLgh = 1;     % gradh and Lgh are unit vectors
                        % Lgh*Lgh.'; or diag(Lgh*Lgh.').' if calculated for multiple time instants
    end
    
    % CBF controller
    function [vsafe,vdes] = K_CBF(z,par)
        alpha = par.alpha;
        % desired control input
        vdes = K_des(z,par);
        % safety filter
        [h,Lfh,Lgh,LghLgh] = CBF(z,par);
        phi = Lfh + sum(Lgh.*vdes.',2).' + alpha*h; % or diag(Lgh*udes).'
        vsafe = vdes + max(0,-phi).*Lgh.'/LghLgh;   % analytical solution of QP
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

    out.tS = tS;
    out.xS = xS;
    out.hS = hS;
    out.tD = tD;
    out.xD = xD;
    out.hD = hD;

end