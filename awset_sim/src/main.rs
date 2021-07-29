use core::f64;
use std::{
    cmp,
    collections::{HashMap, HashSet},
    fmt::Debug,
    usize,
};

use delta_crdt::AWORSet;
use rand::{
    distributions::{Bernoulli, Distribution, Standard, Uniform, WeightedIndex},
    prelude::IteratorRandom,
    Rng,
};
use rand_distr::WeightedAliasIndex;

enum RoundDiff {
    ClassicExclCount,
    ClassicExclSet,
    ProbExclCount,
    ProbExclSet,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum Op {
    Add = 0,
    Remove = 1,
    Sync = 2,
}

// Behaviour of the network topology during the simulation
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum NetworkTopology {
    Static,
    Intermediate,
    Dynamic,
}

#[derive(Clone, Copy, Debug)]
// Effects on the topology when the network is dynamic
enum TopologyEffect {
    Grow,
    Maintain,
    // PermRetire,
    TempRetire,
}

impl Distribution<Op> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Op {
        match rng.gen_range(0..3) {
            0 => Op::Add,
            1 => Op::Remove,
            _ => Op::Sync,
        }
    }
}

// # Configuration
// ## Topology
// const TOPO: NetworkTopology = NetworkTopology::Static;
const TOPO: NetworkTopology = NetworkTopology::Intermediate;
// const TOPO: NetworkTopology = NetworkTopology::Dynamic;
const NUM_REPLICAS: usize = 16;
const LEAVE_PROB: f64 = 0.01; // LEAVE_PROB in [0, 1]
const REVIVE_PROB: f64 = 1.; // REVIVE_PROB in [0, 1]
const RING: bool = false;

// ## Simulation
const NUM_WAVES: u32 = 20_000;
const NUM_ROUNDS: u32 = 3;
const ZIPFIAN_ORIGIN: bool = false;

// ## Synchronization
const CUSTOM_SYNC: bool = true;
const CUSTOM_PROB: u32 = 30; // %
const SYNC_COUNT: u32 = 2_000;
const DELTA_SYNC: bool = false;
const TRANSITIVE: bool = false;
// const DELTA_SYNC: bool = true;
// const TRANSITIVE: bool = true;

fn main() {
    // Vector of CRDT replicas.
    let mut crdts: Vec<delta_crdt::AWORSet<u32, u64>> = Vec::with_capacity(NUM_REPLICAS);
    // Vector with the delta-group for each replica.
    let mut deltas: Vec<delta_crdt::AWORSet<u32, u64>> = Vec::with_capacity(NUM_REPLICAS);
    // Initializing CRDT replicas and delta-groups.
    for i in 0..NUM_REPLICAS {
        crdts.push(delta_crdt::AWORSet::identified_new(i as u32));
        deltas.push(delta_crdt::AWORSet::delta_new());
    }

    // # Random generators
    // ## Origin generator
    let mut origin_gen = rand::thread_rng();
    // ## Operation generator
    let mut op_gen = rand::thread_rng();
    // Instances of operations on the CRDT.
    // The array allows one instance to be randomly sampled when a custom synchronization
    // probability is defined.
    let ops = [Op::Add, Op::Remove, Op::Sync];
    let add_rem_prob = (100 - CUSTOM_PROB) / 2;
    // let op_weighted_dist = WeightedIndex::new([add_rem_prob, add_rem_prob, CUSTOM_PROB]).unwrap();
    let op_weighted_dist =
        WeightedAliasIndex::new(vec![add_rem_prob, add_rem_prob, CUSTOM_PROB]).unwrap();
    // ## Index generator
    let mut idx_gen = rand::thread_rng();
    // ## Element (to add | remove) generator
    let mut elem_gen = rand::thread_rng();
    // ## Delta sync bool generator
    let mut delta_sync_gen = rand::thread_rng();
    // ## Topology effect generator
    // Instances of effects on the network topology.
    // The array allows one instance to be randomly sampled when a dynamic network is in use.
    let topo_effects = [
        TopologyEffect::Grow,
        TopologyEffect::Maintain,
        // TopologyEffect::PermRetire,
        TopologyEffect::TempRetire,
    ];
    // let topo_effect_weighted_dist = WeightedIndex::new([10, 80, 10]).unwrap();
    let topo_effect_weighted_dist = WeightedAliasIndex::new(vec![10, 80, 10]).unwrap();
    let mut topo_effect_gen = rand::thread_rng();
    // ## Leave generator
    // For simulation of nodes leaving and new ones occupying their place.
    let mut leave_gen = rand::thread_rng();
    let leave_dist = Bernoulli::new(LEAVE_PROB).unwrap();
    // let leave = leave_gen.gen_ratio(1, 10);
    let mut leave_count: u32 = 0;
    // ## Revival generator
    // For when the network is dynamic, to determine if one of the retired replicas is revived
    let mut revive_gen = rand::thread_rng();
    let revive_dist = Bernoulli::new(REVIVE_PROB).unwrap();

    // # Global counters
    // ## Error counters
    let mut total_error_state_pcts = [0.0 as f64; NUM_ROUNDS as usize];
    let mut original_error_state_pcts = [0.0 as f64; NUM_ROUNDS as usize];
    let mut sync_error_pcts = [0.0 as f64; NUM_ROUNDS as usize];

    // ## Operation counters
    let mut global_ops_count = [0 as u32; 3];

    // ## Replica counters
    let mut replica_per_round_counts = [0.; NUM_ROUNDS as usize];

    for round in 0..NUM_ROUNDS {
        // The vectors of CRDTs and deltas are truncated in order to reset to the original topology
        // at the beggining of the round. Used when the network is dynamic and can grow above the
        // initial number of replicas.
        crdts.truncate(NUM_REPLICAS);
        deltas.truncate(NUM_REPLICAS);
        // Clear the data in the CRDTs and their respective delta-groups.
        for i in 0..NUM_REPLICAS {
            crdts[i].clear();
            deltas[i].clear();
        }

        // # Indices
        // Indices of the active replicas.
        let mut active_idxs: HashSet<usize> = (0..NUM_REPLICAS).into_iter().collect();
        // Indices of the replicas that have been temporarily retired.
        let mut temp_retired_idxs: HashSet<usize> = HashSet::new();
        // Indices of the replicas that have been permanently retired.
        let mut perm_retired_idxs: HashSet<usize> = HashSet::new();

        // # Round error/inconsistency counters
        // Counter of total inconsistent states during a round.
        let mut total_inconsistent_state_count: u32 = 0;
        let mut original_inconsistent_state_count: u32 = 0;
        // Counters of original inconsistency occurrences.
        let mut classic_excl_len_incons_count: u32 = 0;
        let mut classic_excl_incons_count: u32 = 0;
        let mut prob_excl_len_incons_count: u32 = 0;
        let mut prob_excl_incons_count: u32 = 0;
        // ## Auxiliary
        // Used to indicate the presence of an original inconsistency.
        let mut original_inconsistency = false;

        // Maps that hold the set of different values between the classic and probabilistic
        // structures in the previous round.
        let mut prev_classic_excls: HashMap<usize, HashSet<u64>> = HashMap::new();
        let mut prev_prob_excls: HashMap<usize, HashSet<u64>> = HashMap::new();
        // The number of different elements between the classic and probabilistic structures for each
        // replica in the previous round.
        let mut prev_classic_excl_lens: HashMap<usize, usize> = HashMap::new();
        let mut prev_prob_excl_lens: HashMap<usize, usize> = HashMap::new();

        let mut last_target: usize = 0;
        let mut last_error_pdots = (
            // Local forgotten dots     - Erroneous reinsertion -> +1 in probabilistic
            HashSet::new(),
            // Local false positives    - Erroneous no-op -> +1 in classic
            HashSet::new(),
            // Remote forgotten dots    - Erroneous no-op -> +1 in probabilistic
            HashSet::new(),
            // Remote false positives   - Erroneous removal -> +1 in classic
            HashSet::new(),
        );
        let mut global_error_pdots = (
            // Local forgotten dots     - Erroneous reinsertion -> +1 in probabilistic
            HashSet::new(),
            // Local false positives    - Erroneous no-op -> +1 in classic
            HashSet::new(),
            // Remote forgotten dots    - Erroneous no-op -> +1 in probabilistic
            HashSet::new(),
            // Remote false positives   - Erroneous removal -> +1 in classic
            HashSet::new(),
        );

        // # Round operation counters
        // Operation counter used to check the distribution of operations that generate
        // inconsistencies.
        // tup.0 -> classic exclusive length inconsistency
        // tup.1 -> classic exclusive set inconsistency
        // tup.2 -> probabilistic exclusive length inconsistency
        // tup.3 -> probabilistic exclusive set inconsistency
        let mut orig_incons_op_dist: HashMap<usize, HashMap<Op, (u32, u32, u32, u32)>> =
            HashMap::new();
        // Counters used to check the distribution of operations between the replicas.
        let mut round_ops_per_replica_count: HashMap<usize, (u32, u32, u32, u32)> = HashMap::new();
        // Counters used to check the distribution of the random generation of operations.
        let mut round_ops_count = [0 as u32; 3];
        // Counters used to check the distribution of operations that generate new inconsistencies.
        let mut round_incons_ops_count = [0 as u32; 3];
        // Total removes that have no effect due to there being no elements to remove at a replica.
        let mut null_removes: u32 = 0;
        // Total waves that do not occur due to not having enough active replicas in the network.
        let mut null_waves: u32 = 0;
        // Total syncs that do not occur due to not having enough active replicas in the network.
        let mut null_syncs: u32 = 0;

        let mut classic_excl_pcts = Vec::new();
        let mut prob_excl_pcts = Vec::new();

        let mut wave_count: u32 = 0;

        // Wave - https://www.youtube.com/watch?v=ux6icj_qRMc
        'waves: for wave in 0..NUM_WAVES {
            // 'waves: while round_ops_count[2] < SYNC_COUNT {
            wave_count += 1;
            println!("> WAVE: {}", wave);
            // println!("> WAVE: {}", wave_count - 1);

            // If the network is dynamic, an effect is randomly sampled; otherwise, the network
            // is to remain the same.
            let topo_effect = match TOPO {
                NetworkTopology::Dynamic => {
                    topo_effects[topo_effect_weighted_dist.sample(&mut topo_effect_gen)]
                }
                _ => TopologyEffect::Maintain,
            };

            match topo_effect {
                TopologyEffect::Grow => {
                    if revive_dist.sample(&mut revive_gen) {
                        let temp_ret_idx = temp_retired_idxs.iter().choose(&mut idx_gen);
                        if let Some(idx) = temp_ret_idx {
                            // If there are any temporarily retired CRDTs, one of them returns.
                            let i = *idx;
                            temp_retired_idxs.remove(&i);
                            active_idxs.insert(i);
                        } else {
                            // Otherwise, a new replica is added to the network.
                            let i = crdts.len();
                            crdts.push(delta_crdt::AWORSet::identified_new(i as u32));
                            deltas.push(delta_crdt::AWORSet::delta_new());
                            active_idxs.insert(i);
                        };
                    } else {
                        // Otherwise, a new replica is added to the network.
                        let i = crdts.len();
                        crdts.push(delta_crdt::AWORSet::identified_new(i as u32));
                        deltas.push(delta_crdt::AWORSet::delta_new());
                        active_idxs.insert(i);
                    }
                }
                TopologyEffect::Maintain => {}
                // TopologyEffect::PermRetire => {
                //     let still_alive = all_set.difference(&perm_retired_idxs);
                //     // Select a random index (active or temporarily retired) and add it to the
                //     // vector of permanently retired indices.
                //     let idx_to_die = still_alive.choose(&mut idx_gen);
                //     if let Some(idx) = idx_to_die {
                //         if temp_retired_idxs.contains(idx) {
                //             temp_retired_idxs.remove(idx);
                //         }
                //         let i = *idx;
                //         perm_retired_idxs.insert(i);
                //     }
                // }
                TopologyEffect::TempRetire => {
                    // Select a random active index and add it to the vector of temporarily
                    // retired indices.
                    let idx_to_sleep = active_idxs.iter().choose(&mut idx_gen);
                    if let Some(idx) = idx_to_sleep {
                        let i = *idx;
                        temp_retired_idxs.insert(i);
                        active_idxs.remove(&i);
                    } else {
                        null_waves += 1;
                        assert_eq!(active_idxs.len(), 0);
                        continue 'waves;
                    }
                }
            };

            if TOPO == NetworkTopology::Intermediate {
                // Used to simulate a dynamic network where a certain number of replicas is always
                // present, with a replica occasionally leaving and another new one arriving.
                if leave_dist.sample(&mut leave_gen) {
                    if let Some(&idx) = active_idxs.iter().choose(&mut origin_gen) {
                        active_idxs.remove(&idx);
                        perm_retired_idxs.insert(idx);
                        let new_idx = crdts.len();
                        crdts.push(AWORSet::identified_new(new_idx as u32));
                        deltas.push(AWORSet::delta_new());
                        active_idxs.insert(new_idx);
                        println!("REPLICA {} LEAVING AND {} JOINING", idx, new_idx);
                        leave_count += 1;
                    }
                }
            }

            // Select active replica where the operation will take place.
            let origin;
            if ZIPFIAN_ORIGIN {
                if !active_idxs.is_empty() {
                    let zipf = zipf::ZipfDistribution::new(active_idxs.len(), 1.03).unwrap();
                    // Alternate method
                    // let mut active_idxs_vec = active_idxs.iter().cloned().collect::<Vec<_>>();
                    // active_idxs_vec.sort();
                    // origin = active_idxs_vec[zipf.sample(&mut origin_gen) - 1]

                    if let Some(possible_origin) =
                        active_idxs.iter().nth(zipf.sample(&mut origin_gen) - 1)
                    {
                        origin = *possible_origin;
                    } else {
                        null_waves += 1;
                        assert_eq!(active_idxs.len(), 0);
                        continue 'waves;
                    }
                } else {
                    null_waves += 1;
                    assert_eq!(active_idxs.len(), 0);
                    continue 'waves;
                }
            } else {
                if let Some(&idx) = active_idxs.iter().choose(&mut origin_gen) {
                    origin = idx;
                } else {
                    null_waves += 1;
                    assert_eq!(active_idxs.len(), 0);
                    continue 'waves;
                }
            }

            // Generate random operation.
            let op: Op = if CUSTOM_SYNC {
                ops[op_weighted_dist.sample(&mut op_gen)]
            } else {
                // When a custom probability for sync is not defined, the operations are equally
                // probable to be generated.
                op_gen.gen()
            };

            match op {
                Op::Add => {
                    if let Some(aworset) = crdts.get_mut(origin) {
                        // Register the ocurrence of the operation
                        if let Some((add, _, _, total)) =
                            round_ops_per_replica_count.get_mut(&origin)
                        {
                            *add += 1;
                            *total += 1;
                        } else {
                            round_ops_per_replica_count.insert(origin, (1, 0, 0, 1));
                        }

                        let element: u64 = elem_gen.gen();
                        println!("ADD {} AT {}", element, origin);
                        let delta = aworset.add(element);

                        if DELTA_SYNC {
                            // Merge the delta with the group of previous deltas
                            if let Some(delta_group) = deltas.get_mut(origin) {
                                delta_group.join(&delta);
                            }
                        }
                    }
                }
                Op::Remove => {
                    if let Some(aworset) = crdts.get_mut(origin) {
                        if let Some(&&element) = aworset.read().iter().choose(&mut elem_gen) {
                            // Register the ocurrence of the operation
                            if let Some((_, rem, _, total)) =
                                round_ops_per_replica_count.get_mut(&origin)
                            {
                                *rem += 1;
                                *total += 1;
                            } else {
                                round_ops_per_replica_count.insert(origin, (0, 1, 0, 1));
                            }

                            println!("REMOVE {} AT {}", element, origin);
                            let delta = aworset.remove(element);

                            if DELTA_SYNC {
                                // Merge the delta with the group of previous deltas
                                if let Some(delta_group) = deltas.get_mut(origin) {
                                    delta_group.join(&delta);
                                }
                            }
                        } else {
                            null_removes += 1;
                            continue 'waves;
                        }
                    }
                }
                Op::Sync => {
                    if !(active_idxs.len() > 1) {
                        null_syncs += 1;
                        continue 'waves;
                    };

                    // Register the ocurrence of the operation
                    if let Some((_, _, sync, total)) = round_ops_per_replica_count.get_mut(&origin)
                    {
                        *sync += 1;
                        *total += 1;
                    } else {
                        round_ops_per_replica_count.insert(origin, (0, 0, 1, 1));
                    }

                    let target = loop {
                        if let Some(&possible_target) = active_idxs.iter().choose(&mut idx_gen) {
                            if origin != possible_target {
                                break possible_target;
                            }
                        };
                    };

                    println!("SYNC WITH {} AT {}", target, origin);

                    // This is whole-state and not delta-state sync.
                    let point_break = cmp::min(origin, target) + 1;
                    let (left_crdts, right_crdts) = crdts.split_at_mut(point_break);

                    // If the selected sync method is delta-sync, there is the chance to sync only
                    // the deltas accumulated.
                    // Choose between full-state and delta-group sync (with equal probability)
                    let delta_sync: bool = if DELTA_SYNC {
                        delta_sync_gen.gen_bool(0.5)
                    } else {
                        false
                    };

                    if delta_sync {
                        last_error_pdots = crdts[origin].join(&deltas[target]);
                        if TRANSITIVE {
                            let (d_left, d_right) = deltas.split_at_mut(point_break);
                            if origin >= d_left.len() {
                                last_error_pdots =
                                    d_right[origin - d_left.len()].join(&d_left[target]);
                            } else {
                                last_error_pdots =
                                    d_left[origin].join(&d_right[target - d_left.len()]);
                            }
                        }
                        // Clear the delta-group of the target
                        deltas[target].clear();
                    } else {
                        if origin >= left_crdts.len() {
                            last_error_pdots =
                                right_crdts[origin - left_crdts.len()].join(&left_crdts[target]);
                            // left_crdts[target].join(&right_crdts[origin - left_crdts.len()]);
                        } else {
                            last_error_pdots =
                                left_crdts[origin].join(&right_crdts[target - left_crdts.len()]);
                            // right_crdts[target - left_crdts.len()].join(&left_crdts[origin]);
                        }
                    }

                    global_error_pdots.0.extend(last_error_pdots.0);
                    global_error_pdots.1.extend(last_error_pdots.1);
                    global_error_pdots.2.extend(last_error_pdots.2);
                    global_error_pdots.3.extend(last_error_pdots.3);

                    last_target = target;
                }
            }

            // Register the ocurrence of the operation
            round_ops_count[op as usize] += 1;

            // Checking the consistency between the classic and probabilistic structures and
            // collecting related data.
            let same = crdts[origin].check_eq();
            println!("SAME? = {}", same);
            if !same {
                total_inconsistent_state_count += 1;

                // The type inserted in the AWORSet CRDT is required to be Clone, as specified in
                // the delta_crdt crate used here.
                let classic = crdts[origin].read();
                let prob = crdts[origin].prob_read();
                let classic_excl = classic
                    .difference(&prob)
                    .map(|rr| *rr)
                    .cloned()
                    .collect::<HashSet<_>>();
                let prob_excl = prob
                    .difference(&classic)
                    .map(|rr| *rr)
                    .cloned()
                    .collect::<HashSet<_>>();

                // Closure to register an operation that generated an original inconsistency.
                let mut count_inconsistent_op = |round_diff: RoundDiff| {
                    original_inconsistency = true;
                    // Count the operation as inconsistent for the `origin` replica.
                    if let Some(incons_count_per_op_map) = orig_incons_op_dist.get_mut(&origin) {
                        if let Some(op_incons_count) = incons_count_per_op_map.get_mut(&op) {
                            match round_diff {
                                RoundDiff::ClassicExclCount => (*op_incons_count).0 += 1,
                                RoundDiff::ClassicExclSet => (*op_incons_count).1 += 1,
                                RoundDiff::ProbExclCount => (*op_incons_count).2 += 1,
                                RoundDiff::ProbExclSet => (*op_incons_count).3 += 1,
                            }
                        } else {
                            let v = match round_diff {
                                RoundDiff::ClassicExclCount => (1, 0, 0, 0),
                                RoundDiff::ClassicExclSet => (0, 1, 0, 0),
                                RoundDiff::ProbExclCount => (0, 0, 1, 0),
                                RoundDiff::ProbExclSet => (0, 0, 0, 1),
                            };
                            incons_count_per_op_map.insert(op, v);
                        }
                    } else {
                        let mut op_map = HashMap::new();
                        let v = match round_diff {
                            RoundDiff::ClassicExclCount => (1, 0, 0, 0),
                            RoundDiff::ClassicExclSet => (0, 1, 0, 0),
                            RoundDiff::ProbExclCount => (0, 0, 1, 0),
                            RoundDiff::ProbExclSet => (0, 0, 0, 1),
                        };
                        op_map.insert(op, v);
                        orig_incons_op_dist.insert(origin, op_map);
                    }
                };

                // If the number of elements exclusive to the classic set has changed from its
                // previous value, an additional inconsistency has appeared.
                // What if the size of the exclusive set is the same, but the set is different?
                if let Some(prev_classic_excl_len) = prev_classic_excl_lens.get_mut(&origin) {
                    if *prev_classic_excl_len != classic_excl.len() {
                        // Even if the CRDT is intra-inconsistent and had a previous inconsistency of
                        // this kind recorded, different from the one it currently holds, not every
                        // value will represent a new inconsistency.
                        // classic_excl.len() > *prev_classic_excl_len guarantees that there is a new
                        // inconsistency. However, < could also work, if the set of previous values
                        // is not a superset of the new set.
                        if classic_excl.len() != 0 && classic_excl.len() > *prev_classic_excl_len {
                            classic_excl_len_incons_count += 1;
                            count_inconsistent_op(RoundDiff::ClassicExclCount);
                        }
                        *prev_classic_excl_len = classic_excl.len();
                    }
                } else if classic_excl.len() != 0 {
                    classic_excl_len_incons_count += 1;
                    count_inconsistent_op(RoundDiff::ClassicExclCount);
                    prev_classic_excl_lens.insert(origin, classic_excl.len());
                };
                // If the set of elements exclusive to the classic set has changed, an additional
                // inconsistency has appeared.
                if let Some(prev_classic_excl) = prev_classic_excls.get_mut(&origin) {
                    if *prev_classic_excl != classic_excl {
                        // If the new set of inconsistencies is different and has some value that
                        // was not previously known, a new inconsistency is said to have occurred.
                        if !classic_excl.is_subset(prev_classic_excl) {
                            // Check if every pdot tied to an error results from the expected cause
                            for inconsistent_element in classic_excl.iter() {
                                let possible_pdot =
                                    crdts[origin].pdot_of_value(inconsistent_element);
                                if let Some(pdot) = possible_pdot {
                                    let from_false_positive = global_error_pdots.1.contains(&pdot)
                                        || global_error_pdots.3.contains(&pdot);
                                    assert_eq!(from_false_positive, true);
                                }
                            }
                            classic_excl_incons_count += 1;
                            count_inconsistent_op(RoundDiff::ClassicExclSet);
                        }
                        *prev_classic_excl = classic_excl;
                    }
                } else if !classic_excl.is_empty() {
                    classic_excl_incons_count += 1;
                    count_inconsistent_op(RoundDiff::ClassicExclSet);
                    prev_classic_excls.insert(origin, classic_excl);
                }
                // If the number of elements exclusive to the probabilistic set has changed from its
                // previous value, an additional inconsistency has happened.
                if let Some(prev_prob_excl_len) = prev_prob_excl_lens.get_mut(&origin) {
                    if *prev_prob_excl_len != prob_excl.len() {
                        if prob_excl.len() != 0 && prob_excl.len() > *prev_prob_excl_len {
                            prob_excl_len_incons_count += 1;
                            count_inconsistent_op(RoundDiff::ProbExclCount);
                        }
                        *prev_prob_excl_len = prob_excl.len();
                    }
                } else if prob_excl.len() != 0 {
                    prob_excl_len_incons_count += 1;
                    count_inconsistent_op(RoundDiff::ProbExclCount);
                    prev_prob_excl_lens.insert(origin, prob_excl.len());
                }
                // If the set of elements exclusive to the probabilistic set has changed, an
                // additional inconsistency has happened.
                if let Some(prev_prob_excl) = prev_prob_excls.get_mut(&origin) {
                    if *prev_prob_excl != prob_excl {
                        // If the new set of inconsistencies is different and has some value that
                        // was not previously known, a new inconsistency is said to have occurred.
                        if !prob_excl.is_subset(prev_prob_excl) {
                            // Check if every pdot tied to an error results from the expected cause
                            for inconsistent_element in prob_excl.iter() {
                                let possible_pdot =
                                    crdts[origin].pdot_of_value(inconsistent_element);
                                if let Some(pdot) = possible_pdot {
                                    let from_forgotten_dot = global_error_pdots.0.contains(&pdot)
                                        || global_error_pdots.2.contains(&pdot);
                                    assert!(from_forgotten_dot);
                                }
                            }
                            prob_excl_incons_count += 1;
                            count_inconsistent_op(RoundDiff::ProbExclSet);
                        }
                        *prev_prob_excl = prob_excl;
                    }
                } else if !prob_excl.is_empty() {
                    prob_excl_incons_count += 1;
                    count_inconsistent_op(RoundDiff::ProbExclSet);
                    prev_prob_excls.insert(origin, prob_excl);
                }

                if original_inconsistency {
                    original_inconsistency = false;
                    original_inconsistent_state_count += 1;
                    round_incons_ops_count[op as usize] += 1;
                }
            } else {
                if let Some(prev_classic_excl_len) = prev_classic_excl_lens.get_mut(&origin) {
                    *prev_classic_excl_len = 0;
                }
                if let Some(prev_classic_excl) = prev_classic_excls.get_mut(&origin) {
                    if prev_classic_excl.len() != 0 {
                        *prev_classic_excl = HashSet::new();
                    }
                }
                if let Some(prev_prob_excl_len) = prev_prob_excl_lens.get_mut(&origin) {
                    *prev_prob_excl_len = 0;
                }
                if let Some(prev_prob_excl) = prev_prob_excls.get_mut(&origin) {
                    if prev_prob_excl.len() != 0 {
                        *prev_prob_excl = HashSet::new();
                    }
                }
            }

            if wave % 100 == 0 {
                let classic_excl_set_len =
                    if let Some(prev_classic_excl) = prev_classic_excls.get(&origin) {
                        prev_classic_excl.len()
                    } else {
                        0
                    };
                let prob_excl_set_len = if let Some(prev_prob_excl) = prev_prob_excls.get(&origin) {
                    prev_prob_excl.len()
                } else {
                    0
                };
                let prob_set_len = crdts[origin].prob_read().len();
                let prob_set_len_with_missing_elements = prob_set_len + classic_excl_set_len;
                let classic_excl_pct = if prob_set_len_with_missing_elements as f64 != 0. {
                    (classic_excl_set_len as f64 / prob_set_len_with_missing_elements as f64)
                        * 100_f64
                } else {
                    0.
                };
                let prob_excl_pct = if prob_set_len as f64 != 0. {
                    (prob_excl_set_len as f64 / prob_set_len as f64) * 100_f64
                } else {
                    0.
                };
                classic_excl_pcts.push(classic_excl_pct);
                prob_excl_pcts.push(prob_excl_pct);
            }
        }

        // Information summary of the simulation
        println!("{:-^28}", format!(" SUMMARY | ROUND {} ", round));

        let original_error_state_pct_in_round: f64 =
            // (original_inconsistent_state_count as f64 / NUM_WAVES as f64) * 100_f64;
            (original_inconsistent_state_count as f64 / wave_count as f64) * 100_f64;
        assert_eq!(
            original_inconsistent_state_count,
            round_incons_ops_count[0] + round_incons_ops_count[1] + round_incons_ops_count[2]
        );

        let total_error_state_pct_in_round: f64 =
            // (total_inconsistent_state_count as f64 / NUM_WAVES as f64) * 100_f64;
            (total_inconsistent_state_count as f64 / wave_count as f64) * 100_f64;

        println!(
            "> {: <32} {: >7.4} % = {: >6} / {}",
            "ORIGINAL INCONSISTENT STATE %",
            original_error_state_pct_in_round,
            original_inconsistent_state_count,
            // NUM_WAVES
            wave_count
        );
        println!(
            "> {: <32} {: >7.4} % = {: >6} / {}",
            "TOTAL INCONSISTENT STATE %",
            total_error_state_pct_in_round,
            total_inconsistent_state_count,
            // NUM_WAVES
            wave_count
        );

        // Registering the error percentage for the round.
        original_error_state_pcts[round as usize] = original_error_state_pct_in_round;
        total_error_state_pcts[round as usize] = total_error_state_pct_in_round;

        println!(
            "> {: <32} ADD = {} | REMOVE = {} | SYNC = {} | TOTAL = {}",
            "OP. DIST. BY TYPE",
            round_ops_count[0],
            round_ops_count[1] + null_removes,
            round_ops_count[2] + null_syncs,
            round_ops_count[0]
                + round_ops_count[1]
                + round_ops_count[2]
                + null_removes
                + null_syncs
        );
        assert_eq!(
            null_waves
                + round_ops_count[0]
                + round_ops_count[1]
                + null_removes
                + round_ops_count[2]
                + null_syncs,
            // NUM_WAVES
            wave_count
        );

        println!("> {: <32}", "OP. DIST. BY TYPE + REPLICA");
        let mut ops_count_checker = [0 as u32; 3];
        for i in 0..crdts.len() {
            let rep_op_count = round_ops_per_replica_count
                .get(&i)
                .unwrap_or(&(0_u32, 0_u32, 0_u32, 0_u32));
            ops_count_checker[0] += rep_op_count.0;
            ops_count_checker[1] += rep_op_count.1;
            ops_count_checker[2] += rep_op_count.2;
            println!("{: <34} REPL. {} => {:?}", "", i, rep_op_count);
        }
        assert_eq!(round_ops_count, ops_count_checker);

        println!("> {: <32}", "ORIGINAL INCONSISTENCY COUNTERS");
        println!(
            "{: <34} {: <7} EXCL. => COUNTER = {} | SET = {}",
            "", "CLASSIC", classic_excl_len_incons_count, classic_excl_incons_count
        );
        println!(
            "{: <34} {: <7} EXCL. => COUNTER = {} | SET = {}",
            "", "PROB.", prob_excl_len_incons_count, prob_excl_incons_count
        );
        // Check that increasing the counter of inconsistencies related to the length of the
        // exclusive sets has more restrictive requirements.
        assert!(classic_excl_len_incons_count <= classic_excl_incons_count);
        assert!(prob_excl_len_incons_count <= prob_excl_incons_count);

        // Distribution of operations that generate original inconsistencies per replica
        println!("> {: <32}", "ORIG. INCONSISTENCY OP. DIST.");
        for i in 0..crdts.len() {
            println!("{: <34} REPL. {} =>", "", i);
            for op in [Op::Add, Op::Remove, Op::Sync] {
                let tuple = if let Some(incons_op_dist) = orig_incons_op_dist.get(&i) {
                    if let Some(tup) = incons_op_dist.get(&op) {
                        *tup
                    } else {
                        (0, 0, 0, 0)
                    }
                } else {
                    (0, 0, 0, 0)
                };
                println!("{: <36} {:?} ->", "", op);
                println!(
                    "{: <38} {: <7} EXCL. {: <5} = {}",
                    "", "CLASSIC", "COUNT", tuple.0
                );
                println!(
                    "{: <38} {: <7} EXCL. {: <5} = {}",
                    "", "CLASSIC", "SET", tuple.1
                );
                println!(
                    "{: <38} {: <7} EXCL. {: <5} = {}",
                    "", "PROB", "COUNT", tuple.2
                );
                println!(
                    "{: <38} {: <7} EXCL. {: <5} = {}",
                    "", "PROB", "SET", tuple.3
                );
            }
        }

        println!("DIFF COUNT CLASSIC:\n{:#?}", prev_classic_excl_lens);
        println!("DIFF CLASSIC:\n{:#?}", prev_classic_excls);
        println!("DIFF COUNT PROB:\n{:#?}", prev_prob_excl_lens);
        println!("DIFF PROB:\n{:#?}", prev_prob_excls);

        println!("> {}", "INCONSISTENT % OF STATE PER REPLICA");
        for i in active_idxs.iter() {
            // let classic_excl_set_len = *prev_classic_excl_lens.get(&i).unwrap_or(&0);
            let classic_excl_set_len = if let Some(prev_classic_excl) = prev_classic_excls.get(&i) {
                prev_classic_excl.len()
            } else {
                0
            };
            // let prob_excl_set_len = *prev_prob_excl_lens.get(&i).unwrap_or(&0);
            let prob_excl_set_len = if let Some(prev_prob_excl) = prev_prob_excls.get(&i) {
                prev_prob_excl.len()
            } else {
                0
            };
            let classic_set_len = crdts[*i].read().len();
            let prob_set_len = crdts[*i].prob_read().len();
            let prob_set_len_with_missing_elements = prob_set_len + classic_excl_set_len;
            println!(
                "{:<8} {:>4} => TARGET LENGTH OF CLASSIC: {}",
                "", i, classic_set_len
            );
            println!(
                "{:<18}{: <36} | {: <36}",
                "",
                format!(
                    "CLASSIC: {:.4} % = {}/{}",
                    (classic_excl_set_len as f64 / prob_set_len_with_missing_elements as f64)
                        * 100_f64,
                    classic_excl_set_len,
                    prob_set_len_with_missing_elements
                ),
                format!(
                    "PROB.: {:.4} % = {}/{}",
                    (prob_excl_set_len as f64 / prob_set_len as f64) * 100_f64,
                    prob_excl_set_len,
                    prob_set_len
                )
            );
        }

        println!("> {}", "ORIGINAL INCONSISTENCY % PER OP");
        println!(
            "{:>27} {:>17} | {:<30}",
            "", "% OF ORIG INCONS", "% OF OP COUNT"
        );
        for op in [Op::Add, Op::Remove, Op::Sync] {
            println!(
                "{:<4} {:<6} -> {:>8.4} % = {:>7} / {:<7} | {:>8.4} % = {:>7} / {:<7}",
                "",
                format!("{:?}", op),
                (round_incons_ops_count[op as usize] as f64
                    / original_inconsistent_state_count as f64)
                    * 100_f64,
                round_incons_ops_count[op as usize],
                original_inconsistent_state_count,
                (round_incons_ops_count[op as usize] as f64 / round_ops_count[op as usize] as f64)
                    * 100_f64,
                round_incons_ops_count[op as usize],
                round_ops_count[op as usize]
            );
        }

        sync_error_pcts[round as usize] = (round_incons_ops_count[Op::Sync as usize] as f64
            / round_ops_count[Op::Sync as usize] as f64)
            * 100_f64;

        let count_global_error_pdots = global_error_pdots.0.len()
            + global_error_pdots.1.len()
            + global_error_pdots.2.len()
            + global_error_pdots.3.len();

        println!("> {}", "ORIGINAL INCONSISTENCY % PER OP");
        println!(
            "> 0 -> {:>8.4} % = {:>7} / {:<7}",
            // "> 0 -> {:>8.4} % = {:>7} / {:<7}\n {:#?}",
            (global_error_pdots.0.len() as f64 / count_global_error_pdots as f64) * 100_f64,
            global_error_pdots.0.len(),
            count_global_error_pdots,
            // global_error_pdots.0
        );
        println!(
            "> 1 -> {:>8.4} % = {:>7} / {:<7}",
            // "> 1 -> {:>8.4} % = {:>7} / {:<7}\n {:#?}",
            (global_error_pdots.1.len() as f64 / count_global_error_pdots as f64) * 100_f64,
            global_error_pdots.1.len(),
            count_global_error_pdots,
            // global_error_pdots.1
        );
        println!(
            "> 2 -> {:>8.4} % = {:>7} / {:<7}",
            // "> 2 -> {:>8.4} % = {:>7} / {:<7}\n {:#?}",
            (global_error_pdots.2.len() as f64 / count_global_error_pdots as f64) * 100_f64,
            global_error_pdots.2.len(),
            count_global_error_pdots,
            // global_error_pdots.2
        );
        println!(
            "> 3 -> {:>8.4} % = {:>7} / {:<7}",
            // "> 3 -> {:>8.4} % = {:>7} / {:<7}\n {:#?}",
            (global_error_pdots.3.len() as f64 / count_global_error_pdots as f64) * 100_f64,
            global_error_pdots.3.len(),
            count_global_error_pdots,
            // global_error_pdots.3
        );

        // for i in 0..crdts.len() {
        // for i in active_idxs.iter() {
        //     println!("{} CLASSIC -> {:#?}", i, crdts[*i].read());
        //     println!("{} PROB -> {:#?}", i, crdts[*i].prob_read());
        // }

        println!(
            "SYNC % : {:.4} = {} / {}",
            (round_ops_count[2] as f64 / wave_count as f64) * 100_f64,
            round_ops_count[2],
            wave_count
        );

        println!("NUMBER OF REPLICAS SEEN DURING ROUND: {}", crdts.len());
        replica_per_round_counts[round as usize] = crdts.len() as f64;

        println!("CLASSIC CONSISTENCY PROBE: {:?}", classic_excl_pcts);
        println!(
            "AVG: {} %",
            classic_excl_pcts.iter().sum::<f64>() as f64 / classic_excl_pcts.len() as f64
        );
        println!("PROB CONSISTENCY PROBE: {:?}", prob_excl_pcts);
        println!(
            "AVG: {} %",
            prob_excl_pcts.iter().sum::<f64>() as f64 / prob_excl_pcts.len() as f64
        );
    }

    println!("\n{:-^28}", " FINAL SUMMARY ");
    println!("> ORIGINAL ERROR STATE %s: {:?}", original_error_state_pcts);
    println!(
        "AVG: {} %",
        original_error_state_pcts.iter().sum::<f64>() as f64 / NUM_ROUNDS as f64
    );
    println!("> TOTAL ERROR STATE %s: {:?}", total_error_state_pcts);
    println!(
        "AVG: {} %",
        total_error_state_pcts.iter().sum::<f64>() as f64 / NUM_ROUNDS as f64
    );
    println!("> SYNC ERROR %s: {:?}", sync_error_pcts);
    println!(
        "AVG: {} %",
        sync_error_pcts.iter().sum::<f64>() as f64 / NUM_ROUNDS as f64
    );

    println!("> REPLICAS PER ROUND: {:?}", replica_per_round_counts);
    println!(
        "AVG: {}",
        replica_per_round_counts.iter().sum::<f64>() as f64 / NUM_ROUNDS as f64
    );

    println!(
        "LEAVES -> {} | PER ROUND = {}",
        leave_count,
        leave_count as f64 / NUM_ROUNDS as f64
    );
}
