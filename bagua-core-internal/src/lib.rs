#![allow(clippy::needless_return)]
#[macro_use]
extern crate shadow_rs;
#[macro_use]
extern crate float_cmp;

pub mod comm_ops;
pub mod communicators;
pub mod cuda_utils;
pub mod datatypes;
pub mod events;
pub mod kernels;
pub mod resource_pool;
pub mod telemetry;
pub mod env_var;

use crate::comm_ops::CommOpTrait;
use crate::env_var::{get_rank};
use crate::telemetry::{StatisticalAverage, SCHEDULED_THREAD_POOL, TELEMETRY};
use cpp::cpp;
use datatypes::{BaguaBucket, BaguaTensor};
use events::BaguaEventChannel;
use flume::RecvTimeoutError;
use hashbrown::{HashMap, HashSet};
use std::collections::VecDeque;
use std::fmt::Debug;
use std::sync::Arc;
use std::sync::RwLock;
use std::time::Duration;
use thiserror::Error;

cpp! {{
#include <nccl.h>
#include <stdio.h>
#include <iostream>
#include <bagua_utils.h>
}}

#[derive(Error, Debug)]
pub enum BaguaCoreError {
    #[error("invalid bucket")]
    BucketError(String),
    #[error("invalid tensor")]
    TensorError(String),
    #[error("memory pool error")]
    MemPoolError(#[from] sized_object_pool::SizedPoolError),
    #[error("communication backend error")]
    BackendError(String),
    #[error("communicator error")]
    CommunicatorError(String),
    #[error("telemetry error")]
    TelemetryError(String),
    #[error("serialization error")]
    SerializationSerdeJsonError(#[from] serde_json::Error),
    #[error("internal channel error")]
    InternalChannelError(String),
    #[error("http error")]
    HttpCommunicationError(#[from] ureq::Error),
}

#[derive(Debug)]
pub struct BaguaScheduledCommOp {
    pub bucket: Arc<BaguaBucket>,
    pub ops: Vec<Arc<dyn CommOpTrait + Send + Sync>>,
    pub event_channel: BaguaEventChannel,
}

#[derive(Debug)]
pub struct BaguaCommOpChannels {
    pub schedule_channel_sender: flume::Sender<BaguaScheduledCommOp>,
    pub schedule_channel_receiver: flume::Receiver<BaguaScheduledCommOp>,
    pub not_waited_events_sender: flume::Sender<BaguaEventChannel>,
    pub not_waited_events_receiver: flume::Receiver<BaguaEventChannel>,
    pub post_backward_channel_sender: flume::Sender<BaguaScheduledCommOp>,
    pub post_backward_channel_receiver: flume::Receiver<BaguaScheduledCommOp>,
    pub not_waited_post_backward_events_sender: flume::Sender<BaguaEventChannel>,
    pub not_waited_post_backward_events_receiver: flume::Receiver<BaguaEventChannel>,
}

impl BaguaCommOpChannels {
    pub fn new(cap: usize) -> Self {
        let (sender, receiver) = flume::bounded(cap);
        let (ev_sender, ev_receiver) = flume::unbounded();
        let (post_backward_channel_sender, post_backward_channel_receiver) = flume::bounded(cap);
        let (post_backward_ev_sender, post_backwar_ev_receiver) = flume::unbounded();

        Self {
            schedule_channel_sender: sender,
            schedule_channel_receiver: receiver,
            post_backward_channel_sender,
            post_backward_channel_receiver,
            not_waited_post_backward_events_sender: post_backward_ev_sender,
            not_waited_events_sender: ev_sender,
            not_waited_events_receiver: ev_receiver,
            not_waited_post_backward_events_receiver: post_backwar_ev_receiver,
        }
    }
}

pub fn show_version() {
    shadow!(build);
    eprintln!("project_name: {}", build::PROJECT_NAME);
    eprintln!("is_debug: {}", shadow_rs::is_debug());
    eprintln!("version: {}", build::version());
    eprintln!("tag: {}", build::TAG);
    eprintln!("commit_hash: {}", build::COMMIT_HASH);
    eprintln!("commit_date: {}", build::COMMIT_DATE);
    eprintln!("build_os: {}", build::BUILD_OS);
    eprintln!("rust_version: {}", build::RUST_VERSION);
    eprintln!("build_time: {}", build::BUILD_TIME);
    eprintln!("NCCL version: {}", {
        let mut version = 0i32;
        let version_ptr = &mut version;
        unsafe {
            cpp::cpp!([version_ptr as "int *"]
            { NCCLCHECK(ncclGetVersion(version_ptr)); });
        }
        version
    });
}

#[derive(Debug)]
pub struct BaguaCommBackend {
    ordered_buckets: VecDeque<Arc<BaguaBucket>>,
    /// <tensor_id, bagua_bucket>
    bucket_mapping: HashMap<u64, Arc<BaguaBucket>>,
    channels: Arc<BaguaCommOpChannels>,
    managed_ptrs: HashSet<u64>,
    comm_worker: std::thread::JoinHandle<()>,
    comm_monitor: std::thread::JoinHandle<()>,
    speed_metric: Arc<RwLock<StatisticalAverage>>,
}

impl BaguaCommBackend {
    pub fn schedule_comm(&self, bucket: Arc<BaguaBucket>) -> Result<(), BaguaCoreError> {
        let event_channel = BaguaEventChannel::default();
        self.channels
            .schedule_channel_sender
            .send(BaguaScheduledCommOp {
                ops: {
                    let guard = bucket.inner.lock();
                    guard.comm_ops.clone()
                },
                bucket,
                event_channel: event_channel.clone(),
            })
            .map_err(|e| BaguaCoreError::InternalChannelError(format!("{:?}", e)))?;
        Ok(self
            .channels
            .not_waited_events_sender
            .send(event_channel)
            .map_err(|e| BaguaCoreError::InternalChannelError(format!("{:?}", e)))?)
    }

    fn should_schedule(&self) -> Result<bool, BaguaCoreError> {
        return match self.ordered_buckets.front() {
            None => Err(BaguaCoreError::BackendError(
                "ordered buckets not yet set in comm backend".into(),
            )),
            Some(bucket) => {
                if bucket.ready_for_comm() {
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
        };
    }
}

impl BaguaCommBackend {
    pub fn new(schedule_channel_cap: usize, device_id: usize) -> BaguaCommBackend {
        unsafe {
            cpp::cpp!([device_id as "size_t"]
            { CUDACHECK(cudaSetDevice(device_id)); });
        }

        let channels = Arc::new(BaguaCommOpChannels::new(schedule_channel_cap));
        let channels_clone = channels.clone();
        let (monitor_op_start_channel_sender, monitor_op_start_channel_receiver) =
            flume::unbounded();
        let (monitor_op_finish_channel_sender, monitor_op_finish_channel_receiver) =
            flume::unbounded();

        let speed_metric = Arc::new(RwLock::new(StatisticalAverage::new()));
        let speed_metric_clone = speed_metric.clone();
        let mut log_count = 0;

        BaguaCommBackend {
            ordered_buckets: Default::default(),
            bucket_mapping: Default::default(),
            channels,
            managed_ptrs: Default::default(),
            speed_metric: speed_metric_clone,
            comm_worker: std::thread::spawn(move || {
                unsafe {
                    cpp::cpp!([device_id as "size_t"]
                { CUDACHECK(cudaSetDevice(device_id)); });
                }
                let _span = tracing::span!(tracing::Level::TRACE, "execute_ops");
                let _guard = _span.enter();

                let is_cuda_backend = true;
                let mut speeds = Vec::<(u64, f64)>::new();
                let mut comm_event_queue = VecDeque::<(u64, u64, u64)>::new(); // (bytes, start_event, stop_event)

                loop {
                    let comm_op = channels_clone
                        .schedule_channel_receiver
                        .recv()
                        .expect("cannot receive new comm op");
                    tracing::debug!(
                        "worker received scheduled communication operation {:?}",
                        comm_op
                    );

                    if is_cuda_backend {
                        loop {
                            let event_pair = comm_event_queue.front();
                            if event_pair.is_none() {
                                break;
                            }
                            let (comm_bytes, start, stop) = event_pair.unwrap().clone();
                            let elapsed_time_ms = unsafe {
                                cpp::cpp!([start as "cudaEvent_t", stop as "cudaEvent_t"] -> f32 as "float"
                                {
                                    float milliseconds = 0.;
                                    cudaError_t err = cudaEventElapsedTime(&milliseconds, start, stop);
                                    if (err != cudaSuccess) {
                                        if (err == cudaErrorNotReady) {
                                            return -1.;
                                        }
                                        printf("Failed: Cuda error %s:%d '%s'\n", __FILE__,__LINE__,cudaGetErrorString(err)); exit(EXIT_FAILURE);
                                    }

                                    return milliseconds;
                                })
                            };
                            if elapsed_time_ms < 0. {
                                break;
                            }

                            // println!(
                            //     "comm_bytes={}, elapsed_time_ms={}, speed={}",
                            //     comm_bytes,
                            //     elapsed_time_ms,
                            //     (comm_bytes as f64 / elapsed_time_ms as f64)
                            // );

                            comm_event_queue.pop_front();

                            speeds.push((comm_bytes, elapsed_time_ms as f64));
                        }

                        let (total_comm_bytes, total_elapsed_time_ms) = speeds
                            .clone()
                            .into_iter()
                            .reduce(|lhs, rhs| (lhs.0 + rhs.0, lhs.1 + rhs.1))
                            .unwrap_or((0, 0.));

                        // The statistical error in a too small time range is large, report every 100ms
                        if total_elapsed_time_ms > 100. {
                            let total_elapsed_time_s = total_elapsed_time_ms / 1000.;
                            let total_comm_gb = total_comm_bytes as f64 / 1024_f64.powf(3.); 
                            let gbytes_per_second = total_comm_gb / total_elapsed_time_s;

                            log_count += 1;
                            if log_count % 100 == 0 && (get_rank() == 0 || get_rank() == 8 || get_rank() == 17 || get_rank() == 23) {
                                println!("gbytes_per_second={}, speeds={:?}", gbytes_per_second, speeds);
                            }
                            match speed_metric.write() {
                                Ok(mut speed_metric_lock) => speed_metric_lock.record(gbytes_per_second),
                                Err(err) => {
                                    tracing::error!("{:?}", err)
                                }
                            }

                            speeds.clear();
                        }
                    }

                    let start_event: u64 = if is_cuda_backend {
                        unsafe {
                            cpp::cpp!([] -> u64 as "cudaEvent_t"
                            {
                                cudaEvent_t start;
                                CUDACHECK(cudaEventCreate(&start));
                                CUDACHECK(cudaEventRecord(start));

                                return start;
                            })
                        }
                    } else {
                        0
                    };

                    monitor_op_start_channel_sender.send(comm_op.bucket.clone());
                    for op in &comm_op.ops {
                        op.execute_background_communication(
                            comm_op.bucket.clone(),
                            &channels_clone,
                        );
                    }
                    if is_cuda_backend {
                        let end_event: u64 = unsafe {
                            cpp::cpp!([] -> u64 as "cudaEvent_t"
                            {
                                cudaEvent_t end;
                                CUDACHECK(cudaEventCreate(&end));
                                CUDACHECK(cudaEventRecord(end));

                                return end;
                            })
                        };
                        comm_event_queue.push_back((
                            comm_op.bucket.bytes() as u64,
                            start_event,
                            end_event,
                        ));
                    }

                    tracing::debug!("comm op executed: {:?}", comm_op);
                    comm_op.event_channel.finish();
                    tracing::debug!("comm op marked finished: {:?}", comm_op);
                    monitor_op_finish_channel_sender.send(());
                }
            }),
            comm_monitor: std::thread::spawn(move || loop {
                let op_bucket = monitor_op_start_channel_receiver
                    .recv()
                    .expect("monitor cannot receive next comm op bucket");
                match monitor_op_finish_channel_receiver.recv_timeout(Duration::from_secs(300)) {
                    Ok(_) => {}
                    Err(_) => {
                        panic!("{:?} comm op has not finished for 5 min, panic", op_bucket);
                    }
                }
            }),
        }
    }

    /// calling a second time will overwrite previous buckets
    pub fn register_ordered_buckets(
        &mut self,
        buckets: &[&BaguaBucket],
    ) -> Result<(), BaguaCoreError> {
        self.wait_pending_comm_ops()?;
        self.managed_ptrs.clear();
        self.bucket_mapping.clear();
        self.ordered_buckets.clear();
        for bucket in buckets {
            let bucket = Arc::new((*bucket).clone());
            self.ordered_buckets.push_back(bucket.clone());
            for tensor in &bucket.inner.lock().tensors {
                if self.bucket_mapping.contains_key(&tensor.id)
                    || self.managed_ptrs.contains(&tensor.inner.read().raw.ptr)
                {
                    return Err(BaguaCoreError::TensorError(format!(
                        "duplicated tensor detected, id {}, ptr {}",
                        &tensor.id,
                        &tensor.inner.read().raw.ptr
                    )));
                }
                self.bucket_mapping.insert(tensor.id, bucket.clone());
                self.managed_ptrs.insert(tensor.inner.read().raw.ptr);
            }
        }
        Ok(())
    }

    pub fn mark_communication_ready(
        &mut self,
        tensor: &BaguaTensor,
        ready_cuda_event_ptr: u64,
    ) -> Result<(), BaguaCoreError> {
        tensor.mark_comm_ready(ready_cuda_event_ptr);
        while self.should_schedule()? {
            let bucket = self.ordered_buckets.pop_front().unwrap();
            tracing::debug!("bucket {:?} ready for communication", bucket);
            bucket.reset_comm_ready();
            let bucket_clone = bucket.clone();
            self.ordered_buckets.push_back(bucket);
            self.schedule_comm(bucket_clone)?;
        }
        Ok(())
    }

    pub fn wait_pending_comm_ops(&self) -> Result<usize, BaguaCoreError> {
        let _span = tracing::span!(tracing::Level::TRACE, "wait_pending_comm_ops");
        let _guard = _span.enter();
        let mut num_ev = 0;
        loop {
            let ev = self.channels.not_waited_events_receiver.try_recv();
            match ev {
                Ok(x) => {
                    tracing::debug!("waiting for comm ops event {:?}", x);
                    x.wait();
                    tracing::debug!("comm ops event {:?} finished", x);
                    num_ev += 1;
                }
                Err(_) => return Ok(num_ev),
            }
        }
    }

    pub fn start_upload_telemetry(&self, skip: bool) -> Result<(), BaguaCoreError> {
        SCHEDULED_THREAD_POOL.execute(move || match TELEMETRY.as_ref() {
            None => {}
            Some(x) => {
                let mut guard = x.lock();
                match skip {
                    true => {
                        guard.clear();
                    }
                    false => {
                        match guard.push_payload_and_clear() {
                            Ok(_) => {}
                            Err(x) => {
                                tracing::error!("{:?}", x)
                            }
                        };
                    }
                }
            }
        });
        Ok(())
    }

    pub fn execute_post_backward_comm_ops(&self) -> Result<usize, BaguaCoreError> {
        let mut num_ev = 0;
        loop {
            let comm_op = self.channels.post_backward_channel_receiver.try_recv();
            match comm_op {
                Ok(comm_op) => {
                    tracing::debug!("received post step communication operation {:?}", comm_op);
                    for op in &comm_op.ops {
                        op.execute_background_communication(comm_op.bucket.clone(), &self.channels);
                    }
                    tracing::debug!("comm op executed: {:?}", comm_op);
                    comm_op.event_channel.finish();
                    tracing::debug!("comm op marked finished: {:?}", comm_op);
                    num_ev += 1;
                }
                Err(_) => return Ok(num_ev),
            }
        }
    }

    pub fn wait_pending_post_backward_comm_ops(&self) -> Result<usize, BaguaCoreError> {
        let mut num_ev = 0;
        loop {
            let ev = self
                .channels
                .not_waited_post_backward_events_receiver
                .try_recv();
            match ev {
                Ok(x) => {
                    tracing::debug!("waiting for comm ops event {:?}", x);
                    x.wait();
                    tracing::debug!("comm ops event {:?} finished", x);
                    num_ev += 1;
                }
                Err(_) => return Ok(num_ev),
            }
        }
    }

    pub fn get_speed(&self, last_n_seconds: f64) -> Result<f64, BaguaCoreError> {
        match self.speed_metric.read() {
            Ok(speed_metric) => Ok(speed_metric.get(last_n_seconds)),
            Err(err) => Err(BaguaCoreError::BackendError(format!("{:?}", err))),
        }
    }
}
