#include "leaf/runtime/host_support.h"
#include <iostream>
#include <memory>

int main() {
    try {
        std::size_t cases=0;
        for (unsigned threads : {1,2,4}) {
            leaf::runtime::detail::Workers workers(threads);
            if (workers.thread_count()!=threads) throw std::runtime_error("Wrong thread count");
            for (std::size_t count : {0,1,31,32,33,47,48,49,95,96,97,383,384,385,767,768,769,1535,1536,1537}) {
                for (std::size_t grain : {1,32,48,96}) {
                    auto visits=std::make_unique<std::atomic<unsigned>[]>(count);
                    for(std::size_t i=0;i<count;++i) visits[i]=0;
                    std::atomic<unsigned> invalid{0};
                    workers.run(count,[&](std::size_t first,std::size_t last) {
                        if (first>last || last>count) { ++invalid; return; }
                        for(std::size_t i=first;i<last;++i) ++visits[i];
                    },grain);
                    if(invalid) throw std::runtime_error("Invalid worker range");
                    for(std::size_t i=0;i<count;++i)
                        if(visits[i]!=1) throw std::runtime_error("Omitted or repeated row");
                    ++cases;
                }
            }
            bool rejected=false;
            try { workers.run(1,[](auto,auto){},0); }
            catch(const std::runtime_error&) { rejected=true; }
            if(!rejected) throw std::runtime_error("Zero grain was accepted");
            if constexpr (sizeof(std::size_t)>sizeof(unsigned)) {
                rejected=false;
                try { workers.run(1,[](auto,auto){},std::size_t(std::numeric_limits<unsigned>::max())+1); }
                catch(const std::runtime_error&) { rejected=true; }
                if(!rejected) throw std::runtime_error("Overflowing grain was accepted");
            }
            std::atomic<unsigned> reused{0};
            workers.run(97,[&](auto first,auto last){reused+=unsigned(last-first);});
            if(reused!=97) throw std::runtime_error("Pool failed after rejected grain");
        }
        std::cout<<"Worker grain correctness: "<<cases<<" exactly-once/reuse cases pass\n";
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
