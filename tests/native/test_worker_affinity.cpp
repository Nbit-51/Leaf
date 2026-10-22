#include "leaf/runtime/host_support.h"
#include <chrono>
#include <iostream>
#include <memory>
#include <set>

using namespace leaf::runtime::detail;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

void exercise(unsigned threads, const std::vector<unsigned>& cpus, std::size_t& cases) {
    const auto before = worker_affinity_supported() ? allowed_worker_cpus() : std::vector<unsigned>{};
    {
        Workers workers(threads, cpus);
        if (worker_affinity_supported()) require(allowed_worker_cpus()==before, "Construction changed caller affinity");
        for (std::size_t count : {0,1,31,32,33,63,64,65,95,96,97,127,128,129,383,384,385,1023}) {
            auto visits = std::make_unique<std::atomic<unsigned>[]>(count);
            for (std::size_t i=0; i<count; ++i) visits[i]=0;
            workers.run(count,[&](auto first, auto last) {
                require(first<=last && last<=count, "Invalid task range");
                if (!cpus.empty()) {
                    const auto actual = allowed_worker_cpus();
                    require(actual.size()==1 && std::find(cpus.begin(),cpus.end(),actual[0])!=cpus.end(),
                            "Task did not execute on an assigned CPU");
                }
                for (auto i=first; i<last; ++i) ++visits[i];
            });
            for (std::size_t i=0; i<count; ++i) require(visits[i]==1,"Lost or duplicate output row");
            if (worker_affinity_supported()) require(allowed_worker_cpus()==before,"Run leaked caller affinity");
            ++cases;
        }
        if (!cpus.empty()) {
            {
                auto forward_scope=workers.bind_caller();
                workers.run(96,[&](auto,auto) {
                    require(allowed_worker_cpus().size()==1,"Nested operation lost binding");
                });
                require(allowed_worker_cpus()==std::vector<unsigned>{cpus[0]},"Nested operation restored too early");
                forward_scope.restore();
                require(allowed_worker_cpus()==before,"Forward scope leaked caller affinity");
                ++cases;
            }
            std::atomic<unsigned> arrivals{0}, seen{0};
            const auto caller = std::this_thread::get_id();
            workers.run(32*threads,[&](auto, auto) {
                const auto actual = allowed_worker_cpus();
                require(actual.size()==1,"Assigned thread has a broad mask");
                const auto position = std::find(cpus.begin(),cpus.end(),actual[0]);
                require(position!=cpus.end(),"Unexpected thread CPU");
                if (std::this_thread::get_id()==caller) require(actual[0]==cpus[0],"Caller used another worker CPU");
                seen.fetch_or(1u << unsigned(position-cpus.begin()));
                ++arrivals;
                const auto deadline = std::chrono::steady_clock::now()+std::chrono::seconds(10);
                while (arrivals.load()!=threads) {
                    require(std::chrono::steady_clock::now()<deadline,"Not all owned workers reached the barrier");
                    std::this_thread::yield();
                }
            });
            require(seen==((1u<<threads)-1),"Not every assigned core performed a task");
            require(allowed_worker_cpus()==before,"Barrier leaked caller affinity");
            ++cases;
        }
        bool caught=false;
        try { workers.run(4096,[](auto,auto){ throw std::runtime_error("expected task failure"); }); }
        catch (const std::runtime_error& error) { caught=std::string(error.what())=="expected task failure"; }
        require(caught,"Task exception did not reach caller");
        if (worker_affinity_supported()) require(allowed_worker_cpus()==before,"Exception leaked caller affinity");
        std::atomic<unsigned> visits{0};
        workers.run(4096,[&](auto first,auto last){ visits+=unsigned(last-first); });
        require(visits==4096,"Pool not reusable after exception");
        cases+=2;
    }
    if (worker_affinity_supported()) require(allowed_worker_cpus()==before,"Destruction changed caller affinity");
}

int main() {
    try {
        std::size_t cases=0;
        for (unsigned threads : {1,2,4}) exercise(threads,{},cases);
        if (worker_affinity_supported()) {
            const auto original=allowed_worker_cpus();
            require(!original.empty(),"No allowed CPUs");
            const std::set<unsigned> counts{1,std::min(2u,unsigned(original.size())),std::min(4u,unsigned(original.size()))};
            for (auto threads : counts) {
                std::vector<unsigned> cpus(original.begin(),original.begin()+threads);
                exercise(threads,cpus,cases);
            }
            for (const auto& cpus : std::vector<std::vector<unsigned>>{{original[0],original[0]}, {unbound_cpu}, {original[0],unbound_cpu}, {original[0],original[0],original[0]}}) {
                bool rejected=false;
                try { Workers invalid(2,cpus); }
                catch (const std::runtime_error&) { rejected=true; }
                require(rejected,"Malformed CPU policy was accepted");
                require(allowed_worker_cpus()==original,"Rejected policy changed caller affinity");
                ++cases;
            }
            {
                AffinityScope outer(original[0]);
                require(allowed_worker_cpus()==std::vector<unsigned>{original[0]},"Scoped binding failed");
                AffinityScope inner(original[0]); inner.restore();
                require(allowed_worker_cpus()==std::vector<unsigned>{original[0]},"Nested restoration widened affinity");
                bool rejected=false;
                try { AffinityScope invalid(unbound_cpu-1); }
                catch (const std::runtime_error&) { rejected=true; }
                require(rejected,"Out-of-range CPU was accepted");
            }
            require(allowed_worker_cpus()==original,"Nested binding leaked caller affinity");
        } else {
            bool rejected=false;
            try { Workers invalid(1,{0}); }
            catch (const std::runtime_error&) { rejected=true; }
            require(rejected,"Unsupported placement was silently accepted");
        }
        std::cout << "Worker placement: " << cases << " range, affinity, restoration and exception cases pass\n";
    } catch (const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
