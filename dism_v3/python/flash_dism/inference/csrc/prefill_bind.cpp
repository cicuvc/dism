#include "prefill.hpp"
#include "prefill_training.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py=pybind11;
using namespace dism_prefill;

template<class T> T *training_buffer(const py::dict &buffers,const char *name,int64_t count,
                                    std::vector<std::pair<uintptr_t,uintptr_t>> &ranges) {
    auto a=py::cast<py::array>(buffers[name]);
    if (!a.dtype().is(py::dtype::of<T>()) || a.ndim()!=1 || a.size()!=count
        || !(a.flags()&py::array::c_style) || !a.writeable())
        throw std::invalid_argument(std::string(name)+": writable contiguous 1D buffer with exact dtype/size required");
    auto begin=reinterpret_cast<uintptr_t>(a.mutable_data()),end=begin+count*sizeof(T);
    if (count) {
        for (auto [left,right]:ranges) if (begin<right && left<end)
            throw std::invalid_argument("output buffers must not overlap");
        ranges.emplace_back(begin,end);
    }
    return static_cast<T*>(a.mutable_data());
}

template<class T> py::array_t<T> copy_array(const std::vector<T> &v) {
    py::array_t<T> a(v.size());
    std::copy(v.begin(),v.end(),a.mutable_data());
    return a;
}

template<class T>
py::array_t<T> mock(const Program &p,
    py::array_t<T,py::array::c_style> sq,
    py::array_t<T,py::array::c_style> sk,
    py::array_t<T,py::array::c_style> value) {
    if (sq.ndim()!=2 || sk.ndim()!=2 || value.ndim()!=2 || sq.shape(0)!=p.n
        || sk.shape(0)!=p.n || value.shape(0)!=p.n || sq.shape(1)!=sk.shape(1)
        || sq.shape(1)<=0 || value.shape(1)<=0)
        throw std::invalid_argument("expected sq/sk [N,R], value [N,DV]");
    py::array_t<T> out({py::ssize_t(p.n),value.shape(1)});
    {
        py::gil_scoped_release release;
        execute(p,sq.data(),sk.data(),value.data(),int(sq.shape(1)),int(value.shape(1)),out.mutable_data());
    }
    return out;
}

PYBIND11_MODULE(_dism_prefill,m) {
    py::class_<Program>(m,"Program")
        .def_readonly("n",&Program::n)
        .def("statistics",[](const Program &p) {
            py::dict d;
            d["n"]=p.n; d["states"]=p.states; d["streams"]=p.offsets.size()-1;
            d["events"]=p.rows.size();
            d["program_bytes"]=p.offsets.size()*8+p.rows.size()*20+p.logden.size()*8
                +(p.dlogweight.size()+p.dlogden.size())*8;
            return d;
        })
        .def("arrays",[](const Program &p) {
            py::dict d;
            d["offsets"]=copy_array(p.offsets); d["rows"]=copy_array(p.rows);
            d["decay"]=copy_array(p.decay); d["weight"]=copy_array(p.weight);
            d["logden"]=copy_array(p.logden);
            if (!p.dlogden.empty()) {
                d["dlogweight"]=copy_array(p.dlogweight); d["dlogden"]=copy_array(p.dlogden);
            }
            return d; // Owned copies, never writable aliases to the plan.
        })
        .def("chunks",[](const Program &p,int width) {
            ChunkProgram c;
            { py::gil_scoped_release release; c=chunk_program(p,width); }
            py::dict d;
            d["offsets"]=copy_array(c.offsets); d["rows"]=copy_array(c.rows);
            d["prefixes"]=copy_array(c.prefixes); d["weights"]=copy_array(c.weights);
            d["lengths"]=copy_array(c.lengths); d["resets"]=copy_array(c.resets);
            if (!c.dlog.empty()) d["dlog"]=copy_array(c.dlog);
            return d;
        })
        .def("execute_fp64",&mock<double>,py::arg("sq").noconvert(),py::arg("sk").noconvert(),py::arg("value").noconvert())
        .def("execute_fp32",&mock<float>,py::arg("sq").noconvert(),py::arg("sk").noconvert(),py::arg("value").noconvert());
    m.def("plan",[](const Ids &q,const Ids &k,const Ids &reset,double tau) {
        py::gil_scoped_release release;
        return Builder(q,k,reset,tau).finish();
    });
    // Test oracle: the original binary-lifting LCA path, kept for regression
    // comparison against the O(component) in-component LCA used by plan().
    m.def("plan_reference",[](const Ids &q,const Ids &k,const Ids &reset,double tau) {
        py::gil_scoped_release release;
        return Builder(q,k,reset,tau,true).finish();
    });
    m.def("plan_training",[](const Ids &q,const Ids &k,double tau) {
        py::gil_scoped_release release;
        return Builder(q,k,Ids(q.size(),0),tau,false,true).finish();
    });
    py::class_<TrainingBatchProgram,std::shared_ptr<TrainingBatchProgram>>(m,"TrainingBatchProgram")
        .def("statistics",[](const TrainingBatchProgram &p) {
            py::dict d;
            d["batch"]=p.batch; d["heads"]=p.heads; d["n"]=p.n; d["chunk_size"]=p.width;
            d["tau_rows"]=p.tau_rows;
            d["jobs"]=p.jobs.size(); d["streams"]=p.stream_base.back();
            d["chunks"]=p.chunk_base.back(); d["events"]=p.events; d["native_plan_ms"]=p.plan_ms;
            d["compact"]=p.compact;
            d["multi_streams"]=p.stream_base.back(); d["multi_chunks"]=p.chunk_base.back();
            py::list blocks;
            int64_t single=0;
            for (int k=0;k<3;++k) { blocks.append(p.bucket_blocks[k]); single+=p.bucket_blocks[k]; }
            d["bucket_blocks"]=blocks; d["single_blocks"]=single;
            return d;
        })
        .def("pack_into_compact",[](const TrainingBatchProgram &p,const std::vector<py::dict> &buckets,
                                    const py::dict &multi) {
            if (!p.compact || buckets.size()!=3)
                throw std::invalid_argument("compact plan and three bucket dicts required");
            std::vector<std::pair<uintptr_t,uintptr_t>> ranges;
            std::vector<CompactBucketBuffers> bk(3);
            for (int k=0;k<3;++k) {
                int64_t blocks=p.bucket_blocks[k], slots=blocks*compact_widths[k];
                const auto &d=buckets[k];
                bk[k].rows=training_buffer<int>(d,"rows",slots,ranges);
                bk[k].prefixes=training_buffer<float>(d,"prefixes",slots,ranges);
                bk[k].weights=training_buffer<float>(d,"weights",slots,ranges);
                bk[k].dlog=training_buffer<float>(d,"dlog",slots,ranges);
                bk[k].lengths=training_buffer<int>(d,"lengths",blocks,ranges);
                bk[k].heads=training_buffer<int>(d,"stream_heads",blocks,ranges);
            }
            int64_t streams=p.stream_base.back(),chunks=p.chunk_base.back(),slots=chunks*p.width;
            TrainingBuffers out;
            out.offsets=training_buffer<int64_t>(multi,"offsets",streams+1,ranges);
            out.rows=training_buffer<int>(multi,"rows",slots,ranges);
            out.prefixes=training_buffer<float>(multi,"prefixes",slots,ranges);
            out.weights=training_buffer<float>(multi,"weights",slots,ranges);
            out.dlog=training_buffer<float>(multi,"dlog",slots,ranges);
            out.lengths=training_buffer<int>(multi,"lengths",chunks,ranges);
            out.resets=training_buffer<int>(multi,"resets",chunks,ranges);
            out.stream_heads=training_buffer<int>(multi,"stream_heads",streams,ranges);
            py::gil_scoped_release release;
            p.pack_into_compact(bk,out);
        })
        .def("pack_into",[](const TrainingBatchProgram &p,const py::dict &buffers) {
            int64_t streams=p.stream_base.back(),chunks=p.chunk_base.back(),slots=chunks*p.width;
            std::vector<std::pair<uintptr_t,uintptr_t>> ranges;
            TrainingBuffers out;
            out.offsets=training_buffer<int64_t>(buffers,"offsets",streams+1,ranges);
            out.rows=training_buffer<int>(buffers,"rows",slots,ranges);
            out.prefixes=training_buffer<float>(buffers,"prefixes",slots,ranges);
            out.weights=training_buffer<float>(buffers,"weights",slots,ranges);
            out.dlog=training_buffer<float>(buffers,"dlog",slots,ranges);
            out.lengths=training_buffer<int>(buffers,"lengths",chunks,ranges);
            out.resets=training_buffer<int>(buffers,"resets",chunks,ranges);
            out.stream_heads=training_buffer<int>(buffers,"stream_heads",streams,ranges);
            py::gil_scoped_release release;
            p.pack_into(out);
        });
    using Labels=py::array_t<int,py::array::c_style>;
    using Taus=py::array_t<double,py::array::c_style>;
    using Docs=py::array_t<int64_t,py::array::c_style>;
    using Hard=py::array_t<uint8_t,py::array::c_style>;
    py::class_<PlanningWorkers,std::shared_ptr<PlanningWorkers>>(m,"TrainingPlanner")
        .def(py::init<int>(),py::arg("workers")=8)
        .def("plan",[](std::shared_ptr<PlanningWorkers> pool,Labels labels,Taus tau,Docs docs,Hard hard,int width,bool compact) {
            if (labels.ndim()!=4 || labels.shape(0)!=2 || labels.shape(1)<=0 || labels.shape(2)<=0 || labels.shape(3)<=0
                || labels.shape(1)>INT_MAX || labels.shape(2)>INT_MAX || labels.shape(3)>INT_MAX)
                throw std::invalid_argument("labels must be int32 [2,B,H,N]");
            int b=int(labels.shape(1)),h=int(labels.shape(2)),n=int(labels.shape(3));
            if (int64_t(b)*h>INT_MAX || int64_t(b)*h*n>INT_MAX)
                throw std::invalid_argument("flattened vector rows exceed int32");
            if (!((tau.ndim()==1 && tau.shape(0)==h) || (tau.ndim()==2 && tau.shape(0)==b && tau.shape(1)==h)))
                throw std::invalid_argument("tau must be float64 [H] or [B,H]");
            if (docs.ndim()!=2 || docs.shape(1)!=3 || hard.ndim()!=2 || hard.shape(0)!=b || hard.shape(1)!=h)
                throw std::invalid_argument("documents int64 [D,3]=(batch,start,end), hard uint8 [B,H] required");
            int tau_rows=tau.ndim()==1 ? 1 : b;
            auto label_ptr=labels.data(); auto tau_ptr=tau.data(); auto docs_ptr=docs.data(); auto hard_ptr=hard.data();
            int64_t count=docs.shape(0);
            py::gil_scoped_release release;
            return std::make_shared<TrainingBatchProgram>(std::move(pool),label_ptr,b,h,n,tau_ptr,tau_rows,docs_ptr,count,hard_ptr,width,compact);
        },py::arg("labels").noconvert(),py::arg("tau").noconvert(),py::arg("documents").noconvert(),
          py::arg("hard").noconvert(),py::arg("chunk_size")=16,py::arg("compact")=false);
}
