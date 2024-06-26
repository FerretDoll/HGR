#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <map>
#include <functional>
#ifndef _WIN32
#include <errno.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#include <csignal>
#else
#include <Windows.h>
#include <stdint.h>
#endif

#include "Options.hpp"
#include "VFLib.h"
#include "WindowsTime.h"

using namespace vflib;

#ifndef _WIN32
void sig_handler(int sig) {
  switch (sig) {
    case SIGKILL:
      std::cout << "Killed \n";
      exit(-1);
    case SIGABRT:
      std::cout << "Aborted \n";
      exit(-1);
    case SIGTERM:
      std::cout << "Terminated \n";
      exit(-1);
    case SIGSEGV:
      std::cout << "Segmentation fault \n";
      exit(-1);
    case SIGSTOP:
      std::cout << "Stopped \n";
      exit(-1);
    case SIGCHLD:
      std::cout << "Child stopped or terminated \n";
      exit(-1);
    default:
      std::cout << "Unexpected signal " << sig << "\n";
      exit(-1);
  }
}
#endif

#ifdef TRACE
std::string& remove_chars(std::string& s, const std::string& chars) {
  s.erase(remove_if(s.begin(), s.end(),
                    [&chars](const char& c) {
                      return chars.find(c) != std::string::npos;
                    }),
          s.end());
  return s;
}
#endif

static long long state_counter = 0;

template <typename Node, typename Edge, typename StateType>
void solve(Options& opt, std::istream& graphInPat, std::istream& graphInTarg,
           struct timeval& start) {
  struct timeval loading, fastcheck, classification, iter, end;
  uint32_t n1, n2;
  double timeAll = 0;
  double totalExecTime = 0;
  double timeFirst = 0;
  double timeLast = 0;
  double timeLoad = 0;
  int rep = 0;
  std::vector<MatchingSolution> solutions;
  std::vector<uint32_t> class_patt;
  std::vector<uint32_t> class_targ;
  uint32_t classes_count;

  state_counter = 0;
  size_t sols = 0;

  ARGLoader<Node, Edge>* pattloader = CreateLoader<Node, Edge>(opt, graphInPat);
  ARGLoader<Node, Edge>* targloader =
      CreateLoader<Node, Edge>(opt, graphInTarg);

  ARGraph<Node, Edge> patt_graph(pattloader);
  ARGraph<Node, Edge> targ_graph(targloader);

  if (opt.verbose) {
    gettimeofday(&loading, NULL);
    timeLoad = GetElapsedTime(start, loading);
    std::cout << "Loaded in: " << timeLoad << std::endl;
  }

  n1 = patt_graph.NodeCount();
  n2 = targ_graph.NodeCount();

  MatchingEngine<StateType>* me = CreateMatchingEngine<StateType>(opt);

  if (!me) {
    exit(-1);
  }

  gettimeofday(&start, NULL);
  FastCheck<Node, Node, Edge, Edge> check(&patt_graph, &targ_graph);
  if (check.CheckSubgraphIsomorphism()) {
    if (opt.verbose) {
      gettimeofday(&fastcheck, NULL);
      timeLoad = GetElapsedTime(start, fastcheck);
      std::cout << "FastCheck in: " << timeLoad << std::endl;
    }

    gettimeofday(&start, NULL);
    NodeClassifier<Node, Edge> classifier(&targ_graph);
    NodeClassifier<Node, Edge> classifier2(&patt_graph, classifier);
    class_patt = classifier2.GetClasses();
    class_targ = classifier.GetClasses();
    classes_count = classifier.CountClasses();

    if (opt.verbose) {
      gettimeofday(&classification, NULL);
      timeLoad = GetElapsedTime(start, classification);
      std::cout << "Classification in: " << timeLoad << std::endl;
    }
  }

#ifndef TRACE
  do {
    rep++;
    me->ResetSolutionCounter();

    gettimeofday(&iter, NULL);
#endif
    if (check.CheckSubgraphIsomorphism()) {
      VF3NodeSorter<Node, Edge, SubIsoNodeProbability<Node, Edge> > sorter(
          &targ_graph);
      std::vector<nodeID_t> sorted = sorter.SortNodes(&patt_graph);

#ifdef TRACE
      me->InitTrace(outfilename);
#endif

      StateType s0(&patt_graph, &targ_graph, class_patt.data(),
                   class_targ.data(), classes_count, sorted.data());
      me->FindAllMatchings(s0);
#ifdef TRACE
      me->FlushTrace();
#endif
    }

#ifndef TRACE
    gettimeofday(&end, NULL);

    totalExecTime += GetElapsedTime(iter, end);
    end = me->GetFirstSolutionTime();
    timeFirst += GetElapsedTime(iter, end);
  } while (totalExecTime < opt.repetitionTimeLimit);
  timeAll = totalExecTime / rep;
  timeFirst /= rep;

  if (opt.storeSolutions) {
    me->GetSolutions(solutions);
    std::cout << "Solution Found" << std::endl;
    std::vector<MatchingSolution>::iterator it;
    for (it = solutions.begin(); it != solutions.end(); it++) {
      std::cout << me->SolutionToString(*it) << std::endl;
    }
  }

#endif
  sols = me->GetSolutionsCount();
  if (opt.verbose) {
    std::cout << "First Solution in: " << timeFirst << std::endl;
    std::cout << "Matching Finished in: " << timeAll << std::endl;
    std::cout << "Solutions: " << sols << std::endl;

  } else {
    std::cout << sols << " " << timeFirst << " " << timeAll << std::endl;
  }
  delete me;
  delete pattloader;
  delete targloader;
}

int32_t main(int32_t argc, char** argv) {
  Options opt;
  struct timeval start;

#ifndef _WIN32
  std::signal(SIGKILL, sig_handler);
  std::signal(SIGABRT, sig_handler);
  std::signal(SIGTERM, sig_handler);
  std::signal(SIGSEGV, sig_handler);
  std::signal(SIGSTOP, sig_handler);
  std::signal(SIGCHLD, sig_handler);
#endif

  if (!GetOptions(opt, argc, argv)) {
    exit(-1);
  }

#ifdef TRACE
  int status;
  std::string pattern_name(opt.pattern);
  std::string target_name(opt.target);

  pattern_name = pattern_name.substr(pattern_name.find_last_of("/\\") + 1);
  target_name = target_name.substr(target_name.find_last_of("/\\") + 1);

  pattern_name = remove_chars(pattern_name, "./");
  target_name = remove_chars(target_name, "./");

  std::string outpath(argv[3]);

  // mkdir -p foo/bar/zoo/andsoforth
  status = mkdir(outpath.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  // std::cout<<status<<" "<<errno<<std::endl;
  std::string outfilename = outpath + "/" + pattern_name + target_name + ".cvs";
  // std::cout << outfilename;
#endif

  gettimeofday(&start, NULL);

  std::ifstream graphInPat(opt.pattern);
  std::ifstream graphInTarg(opt.target);
  if (opt.format == "vfe") {
    solve<data_t, data_t, state_t>(opt, graphInPat, graphInTarg, start);
  } else {
    solve<data_t, Empty, noedgelabel_state_t>(opt, graphInPat, graphInTarg,
                                              start);
  }

  return 0;
}

// 定义一个类型为void(const char*)的回调函数类型
typedef void (*SolutionCallback)(const char*);

template <typename Node, typename Edge, typename StateType>
void solve_for_python(Options& opt, std::istream& graphInPat, std::istream& graphInTarg,
           struct timeval& start, SolutionCallback solutionCallback) {
  uint32_t n1, n2;
  ARGLoader<Node, Edge>* pattloader = CreateLoader<Node, Edge>(opt, graphInPat);
  ARGLoader<Node, Edge>* targloader = CreateLoader<Node, Edge>(opt, graphInTarg);

  ARGraph<Node, Edge> patt_graph(pattloader);
  ARGraph<Node, Edge> targ_graph(targloader);

  n1 = patt_graph.NodeCount();
  n2 = targ_graph.NodeCount();

  MatchingEngine<StateType>* me = CreateMatchingEngine<StateType>(opt);
  if (!me) {
    exit(-1);
  }

  FastCheck<Node, Node, Edge, Edge> check(&patt_graph, &targ_graph);
  if (check.CheckSubgraphIsomorphism()) {
    NodeClassifier<Node, Edge> classifier(&targ_graph);
    NodeClassifier<Node, Edge> classifier2(&patt_graph, classifier);
    std::vector<uint32_t> class_patt = classifier2.GetClasses();
    std::vector<uint32_t> class_targ = classifier.GetClasses();
    uint32_t classes_count = classifier.CountClasses();

    VF3NodeSorter<Node, Edge, SubIsoNodeProbability<Node, Edge>> sorter(&targ_graph);
    std::vector<nodeID_t> sorted = sorter.SortNodes(&patt_graph);

    StateType s0(&patt_graph, &targ_graph, class_patt.data(), class_targ.data(), classes_count, sorted.data());
    me->FindAllMatchings(s0);
  }

  std::vector<MatchingSolution> solutions;
  if (opt.storeSolutions) {
    me->GetSolutions(solutions);
  }

  // Convert solutions to string and pass to callback
  std::string allSolutionsStr;
  for (auto& solution : solutions) {  // Use non-const reference
    if (!allSolutionsStr.empty()) {
      allSolutionsStr += "|";  // Use '|' as delimiter
    }
    allSolutionsStr += me->SolutionToString(solution);
  }

  // Pass all solutions as a string to the callback, if no solutions found pass an empty string
  solutionCallback(allSolutionsStr.empty() ? "" : allSolutionsStr.c_str());

  delete me;
  delete pattloader;
  delete targloader;
}


void parse_options(const std::string& options_str, Options& opt) {
    std::istringstream iss(options_str);
    std::string token;
    std::map<std::string, std::function<void()>> handlers{
        {"-u", [&]() { opt.undirected = true; }},
        {"-s", [&]() { opt.storeSolutions = true; }},
        {"-v", [&]() { opt.verbose = true; }},
    };

    std::map<std::string, std::function<void(std::string)>> arg_handlers{
        {"-r", [&](std::string arg) { opt.repetitionTimeLimit = std::stof(arg); }},
        {"-f", [&](std::string arg) { opt.format = arg; }},
    };

    while (iss >> token) {
        if (handlers.find(token) != handlers.end()) {
            handlers[token]();  // Call the handler without argument
        } else if (arg_handlers.find(token) != arg_handlers.end()) {
            std::string arg;
            if (!(iss >> arg)) {
                std::cerr << "Expected argument after " << token << std::endl;
                return;
            }
            arg_handlers[token](arg);  // Call the handler with argument
        } else {
            std::cerr << "Unknown option: " << token << std::endl;
        }
    }
}

extern "C" {
    void run_vf3(const char* pattern_data, const char* target_data, const char* options, SolutionCallback callback) {
        Options opt;
        parse_options(options, opt);

        std::istringstream graphInPat(pattern_data);
        std::istringstream graphInTarg(target_data);
        struct timeval start;
        gettimeofday(&start, NULL);

        // 根据选项决定使用哪种数据和状态类型
        if (opt.format == "vfe") {
            solve_for_python<data_t, data_t, state_t>(opt, graphInPat, graphInTarg, start, callback);
        } else {
            solve_for_python<data_t, Empty, noedgelabel_state_t>(opt, graphInPat, graphInTarg, start, callback);
        }
    }
}

