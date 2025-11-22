#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/lte-module.h"
#include "ns3/point-to-point-helper.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include <iostream>
#include <iomanip>

using namespace ns3;

static double MbpsToBps(double mbps) { return mbps * 1e6; }

int main (int argc, char *argv[])
{
  double ue1 = 1.0, ue2 = 1.0, ue3 = 1.0; // Mb/s requested (downlink)
  double duration = 60.0;                 // seconds

  CommandLine cmd;
  cmd.AddValue("ue1", "UE1 downlink Mbps", ue1);
  cmd.AddValue("ue2", "UE2 downlink Mbps", ue2);
  cmd.AddValue("ue3", "UE3 downlink Mbps", ue3);
  cmd.AddValue("duration", "Simulation time (s)", duration);
  cmd.Parse(argc, argv);

  // Core helpers
  Ptr<LteHelper> lteHelper = CreateObject<LteHelper>();
  Ptr<PointToPointEpcHelper> epcHelper = CreateObject<PointToPointEpcHelper>();
  lteHelper->SetEpcHelper(epcHelper);
  lteHelper->SetSchedulerType("ns3::RrFfMacScheduler"); // simple

  Ptr<Node> pgw = epcHelper->GetPgwNode();

  // Create remote host (traffic source server side)
  NodeContainer remoteHostContainer;
  remoteHostContainer.Create(1);
  InternetStackHelper internet;
  internet.Install(remoteHostContainer);

  // PGW <-> remoteHost
  PointToPointHelper p2ph;
  p2ph.SetDeviceAttribute("DataRate", StringValue("10Gbps"));
  p2ph.SetChannelAttribute("Delay", StringValue("2ms"));
  NetDeviceContainer internetDevices = p2ph.Install(pgw, remoteHostContainer.Get(0));
  Ipv4AddressHelper ipv4h;
  ipv4h.SetBase("1.0.0.0", "255.0.0.0");
  Ipv4InterfaceContainer internetIfs = ipv4h.Assign(internetDevices);
  Ipv4Address remoteHostAddr = internetIfs.GetAddress(1);

  Ipv4StaticRoutingHelper ipv4RoutingHelper;
  Ptr<Ipv4StaticRouting> remoteHostStaticRouting = ipv4RoutingHelper.GetStaticRouting(remoteHostContainer.Get(0)->GetObject<Ipv4>());
  remoteHostStaticRouting->AddNetworkRouteTo(Ipv4Address("7.0.0.0"), Ipv4Mask("255.0.0.0"), 1);

  // eNB + UEs
  NodeContainer enbNodes; enbNodes.Create(1);
  NodeContainer ueNodes;  ueNodes.Create(3);
  internet.Install(ueNodes);

  // Position (static)
  MobilityHelper mobility;
  mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
  mobility.Install(enbNodes);
  mobility.Install(ueNodes);

  NetDeviceContainer enbDevs = lteHelper->InstallEnbDevice(enbNodes);
  NetDeviceContainer ueDevs  = lteHelper->InstallUeDevice(ueNodes);

  // Assign IP to UEs
  Ipv4InterfaceContainer ueIfs = epcHelper->AssignUeIpv4Addresses(NetDeviceContainer(ueDevs));
  for (uint32_t i=0; i<ueNodes.GetN(); ++i) {
    Ptr<Ipv4StaticRouting> ueStaticRouting = ipv4RoutingHelper.GetStaticRouting(ueNodes.Get(i)->GetObject<Ipv4>());
    ueStaticRouting->SetDefaultRoute(epcHelper->GetUeDefaultGatewayAddress(), 1);
  }
  lteHelper->Attach(ueDevs.Get(0), enbDevs.Get(0));
  lteHelper->Attach(ueDevs.Get(1), enbDevs.Get(0));
  lteHelper->Attach(ueDevs.Get(2), enbDevs.Get(0));

  // Downlink UDP servers on UEs (one per UE)
  uint16_t basePort = 4000;
  for (uint32_t i=0; i<ueNodes.GetN(); ++i) {
    UdpServerHelper srv(basePort + i);
    ApplicationContainer apps = srv.Install(ueNodes.Get(i));
    apps.Start(Seconds(0.1));
    apps.Stop(Seconds(duration));
  }

  // Downlink UDP clients on remote host (send to UEs)
  auto addClient = [&](double rateMbps, Ipv4Address dst, uint16_t port) {
    UdpClientHelper cl(dst, port);
    cl.SetAttribute("Interval", TimeValue(MilliSeconds(10)));
    cl.SetAttribute("MaxPackets", UintegerValue(0xFFFFFFFF));
    cl.SetAttribute("PacketSize", UintegerValue(1200));
    ApplicationContainer cApp = cl.Install(remoteHostContainer.Get(0));
    // Configure rate via OnOff wrapper (simpler to cap)
    OnOffHelper onoff("ns3::UdpSocketFactory", InetSocketAddress(dst, port));
    onoff.SetAttribute("DataRate", DataRateValue(DataRate(static_cast<uint64_t>(MbpsToBps(rateMbps)))));
    onoff.SetAttribute("PacketSize", UintegerValue(1200));
    ApplicationContainer oApp = onoff.Install(remoteHostContainer.Get(0));
    cApp.Start(Seconds(0.2));
    oApp.Start(Seconds(0.2));
    cApp.Stop(Seconds(duration));
    oApp.Stop(Seconds(duration));
  };

  addClient(ue1, ueIfs.GetAddress(0), basePort+0);
  addClient(ue2, ueIfs.GetAddress(1), basePort+1);
  addClient(ue3, ueIfs.GetAddress(2), basePort+2);

  // FlowMonitor for stats
  FlowMonitorHelper fmHelper;
  Ptr<FlowMonitor> monitor = fmHelper.InstallAll();

  Simulator::Stop(Seconds(duration + 0.5));
  Simulator::Run();

  // Collect simple per-UE DL stats
  std::vector<double> thr(3,0.0), loss(3,0.0);
  monitor->CheckForLostPackets();
  auto stats = monitor->GetFlowStats ();

  // Map flows to UE indices by dest IP
  std::map<Ipv4Address,int> ip2idx = {
    { ueIfs.GetAddress(0), 0 },
    { ueIfs.GetAddress(1), 1 },
    { ueIfs.GetAddress(2), 2 }
  };

  for (auto &p : stats) {
    const Ipv4FlowClassifier::FiveTuple t = fmHelper.GetClassifier()->FindFlow(p.first);
    auto it = ip2idx.find(t.destinationAddress);
    if (it == ip2idx.end()) continue;
    int idx = it->second;
    double rxBytes = p.second.rxBytes;
    double txPkts  = p.second.txPackets;
    double rxPkts  = p.second.rxPackets;
    double durSec  = (p.second.timeLastRxPacket - p.second.timeFirstTxPacket).GetSeconds();
    if (durSec <= 0) durSec = duration;
    thr[idx]  += (rxBytes * 8.0) / durSec / 1e6; // Mbps
    double lost = txPkts > 0 ? (100.0 * (txPkts - rxPkts) / txPkts) : 0.0;
    loss[idx] = std::max(loss[idx], lost);
  }

  // Output JSON summary to stdout
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "{"
            << "\"thr\":[" << thr[0] << "," << thr[1] << "," << thr[2] << "],"
            << "\"loss\":[" << loss[0] << "," << loss[1] << "," << loss[2] << "]"
            << "}\n";

  Simulator::Destroy();
  return 0;
}
