# Annotated bibliography: predicting plasticity in amorphous solids at lower compute cost

Companion to [ANALYSIS.md](ANALYSIS.md) (Sect. 6). Every entry below was checked against at least one search hit or repository record with a matching title and venue (arXiv, APS, HAL, PMC, publisher, or the authors' site). **[unconfirmed]** marks any detail I could not check directly.

---

## A. Structural indicators of plasticity and their benchmarks

1. **Richard, D., Ozawa, M., Patinet, S., Stanifer, E., Shang, B., Ridout, S. A., Xu, B., Zhang, G., Morse, P. K., Barrat, J.-L., Berthier, L., Falk, M. L., Guan, P., Liu, A. J., Martens, K., Sastry, S., Vandembroucq, D., Lerner, E., Manning, M. L.** "Predicting plasticity in disordered solids from structural indicators." *Phys. Rev. Materials* **4**, 113609 (2020). DOI: 10.1103/PhysRevMaterials.4.113609. arXiv:2003.11629.
   - This is the main benchmark. It ranks about 20 indicators against the first and later AQS plastic events in 2D binary LJ (N = 10⁴, 100 samples) and in polydisperse glasses. Indicators built from vibrational modes reach C_min close to 1, matching the residual plastic strength Δτ_y (that is, LYS). Most purely structural indicators do poorly; steric bond order Θ and ML softness are the exceptions.

2. **Patinet, S., Vandembroucq, D., Falk, M. L.** "Connecting local yield stresses with plastic activity in amorphous solids." *Phys. Rev. Lett.* **117**, 045501 (2016). DOI: 10.1103/PhysRevLett.117.045501.
   - This paper introduced the frozen-matrix LYS method. Low local yield stress predicts where plastic activity occurs better than "conventional structural properties," and it stays predictive for 50 or more events. (I could not open the full text, so the exact list of comparison indicators is **[unconfirmed]**.)

3. **Barbot, A., Lerbinger, M., Hernandez-Garcia, A., García-García, R., Falk, M. L., Vandembroucq, D., Patinet, S.** "Local yield stress statistics in model amorphous solids." *Phys. Rev. E* **97**, 033001 (2018). DOI: 10.1103/PhysRevE.97.033001. arXiv:1802.05796.
   - This extends LYS to the 2D binary LJ glass we reproduce. The LYS–plasticity correlation is stronger in better-relaxed glasses, local thresholds rise with relaxation, and stress drops follow exponential distributions.

4. **Lerbinger, M., Barbot, A., Vandembroucq, D., Patinet, S.** "Relevance of shear transformations in the relaxation of supercooled liquids." *Phys. Rev. Lett.* **129**, 195501 (2022). DOI: 10.1103/PhysRevLett.129.195501. arXiv:2109.12639.
   - Local residual plastic strengths along soft directions, computed in the inherent structure, correlate with thermal relaxation in supercooled liquids. The correlation is largest near τ_α. This shows LYS is useful beyond athermal shear.

5. **Ruan, D., Patinet, S., Falk, M. L.** "Predicting plastic events and quantifying the local yield surface in 3D model glasses." *J. Mech. Phys. Solids* **158**, 104671 (2022). DOI: 10.1016/j.jmps.2021.104671.
   - LYS in 3D works best with a probe radius of about 5σ. The correlation stays positive for about 200 events (about 7 % strain), and the full local yield surface is anisotropic. Covering several loading directions multiplies the cost of LYS, which is why cheaper proxies are worth having.

6. **Manning, M. L., Liu, A. J.** "Vibrational modes identify soft spots in a sheared disordered packing." *Phys. Rev. Lett.* **107**, 108302 (2011). DOI: 10.1103/PhysRevLett.107.108302.
   - Low-frequency harmonic modes locate "soft spots" where AQS rearrangements start. This is the basis for our Hessian soft-mode proxy.

7. **Cubuk, E. D., Schoenholz, S. S., Rieser, J. M., Malone, B. D., Rottler, J., Durian, D. J., Kaxiras, E., Liu, A. J.** "Identifying structural flow defects in disordered solids using machine-learning methods." *Phys. Rev. Lett.* **114**, 108001 (2015). DOI: 10.1103/PhysRevLett.114.108001.
   - An SVM over radial and angular structure functions ("softness") flags particles likely to rearrange in LJ glasses and granular pillars.

8. **Schoenholz, S. S., Cubuk, E. D., Sussman, D. M., Kaxiras, E., Liu, A. J.** "A structural approach to relaxation in glassy liquids." *Nature Physics* **12**, 469–471 (2016). DOI: 10.1038/nphys3644.
   - Softness correlates strongly with dynamics, and the onset of glassy dynamics coincides with the onset of the softness–dynamics correlation.

9. **Ding, J., Patinet, S., Falk, M. L., Cheng, Y., Ma, E.** "Soft spots and their structural signature in a metallic glass." *PNAS* **111**, 14052–14056 (2014). DOI: 10.1073/pnas.1412095111.
   - In Cu64Zr36, soft spots defined by quasi-localized low-frequency modes match geometrically unfavoured motifs.

10. **Ding, J., Cheng, Y.-Q., Sheng, H., Asta, M., Ritchie, R. O., Ma, E.** "Universal structural parameter to quantitatively predict metallic glass properties." *Nature Communications* **7**, 13733 (2016). DOI: 10.1038/ncomms13733.
    - This paper introduces "flexibility volume," which is atomic volume times vibrational MSD. It predicts the shear modulus and correlates with the propensity for shear transformations.

11. **Xu, B., Falk, M. L., Li, J. F., Kong, L. T.** "Predicting shear transformation events in metallic glasses." *Phys. Rev. Lett.* **120**, 125503 (2018). DOI: 10.1103/PhysRevLett.120.125503.
    - Sampling the energy landscape (saddles and minimum-energy paths), then extrapolating the stress gradient along each path, predicts the triggering strain, location and structure of STZs. Their predictions agree with AQS. This is the closest precedent for our soft-mode saddle-node estimates.

12. **Gartner, L., Lerner, E.** "Nonlinear modes disentangle glassy and Goldstone modes in structural glasses." *SciPost Phys.* **1**, 016 (2016). DOI: 10.21468/SciPostPhys.1.2.016.
    - Nonlinear (cubic) modes avoid hybridising with phonons, so they give a cleaner definition of soft glassy modes than harmonic eigenvectors. This bears on the noise in our Hessian proxy.

13. **Kapteijns, G., Richard, D., Lerner, E.** "Nonlinear quasilocalized excitations in glasses. I. True representatives of soft spots." *Phys. Rev. E* **101**, 032130 (2020). DOI: 10.1103/PhysRevE.101.032130 **[DOI inferred from volume/article number]**. arXiv:1912.10930.
    - Nonlinear quasilocalized excitations represent soft spots robustly and give access to the statistics of energy barriers.

14. **Richard, D., Kapteijns, G., Giannini, J. A., Manning, M. L., Lerner, E.** "Simple and broadly applicable definition of shear transformation zones." *Phys. Rev. Lett.* **126**, 015501 (2021). DOI: 10.1103/PhysRevLett.126.015501.
    - This defines STZs from the harmonic approximation alone (with an open-source library). It is a cheap Hessian-based indicator closely related to our soft-mode proxy.

15. **Ridout, S. A., Rocks, J. W., Liu, A. J.** "Correlation of plastic events with local structure in jammed packings across spatial dimensions." *PNAS* (2022), e2119006119. DOI: 10.1073/pnas.2119006119 (volume 119 **[unconfirmed]**). arXiv:2011.13049.
    - Softness, and even the bare coordination number Z, predict AQS rearrangements in d = 2–5. Simple local features can carry most of the signal.

16. **Widmer-Cooper, A., Harrowell, P.** "Predicting the long-time dynamic heterogeneity in a supercooled liquid on the basis of short-time heterogeneities." *Phys. Rev. Lett.* **96**, 185701 (2006). DOI: 10.1103/PhysRevLett.96.185701. arXiv:cond-mat/0512035.
    - The local Debye–Waller factor predicts dynamic propensity, but **inherent-structure potential energy per particle correlates poorly** with it. This is an early negative result for per-atom energy.

17. **Shi, Y., Falk, M. L.** "Strain localization and percolation of stable structure in amorphous solids." *Phys. Rev. Lett.* **95**, 095502 (2005). DOI: 10.1103/PhysRevLett.95.095502.
    - Whether flow localises is controlled by the potential energy per atom before loading, together with the percolation of short-range order. Energy works as a sample-level stability measure, even if its per-atom predictive power is weak.

## B. Machine learning for glassy dynamics and plasticity

18. **Bapst, V., Keck, T., Grabska-Barwińska, A., Donner, C., Cubuk, E. D., Schoenholz, S. S., et al.** "Unveiling the predictive power of static structure in glassy systems." *Nature Physics* **16**, 448–454 (2020). DOI: 10.1038/s41567-020-0842-8.
    - A GNN on raw positions predicts propensity across temperatures and pressures, and also predicts rearrangements under shear. The dataset has 3D Kob–Andersen systems of N = 4096, with 800 configurations per state point (about 400 for training), each labelled by 30 isoconfigurational runs. These counts come from later papers that reuse the dataset **[secondary source]**.

19. **Fan, Z., Ma, E.** "Predicting orientation-dependent plastic susceptibility from static structure in amorphous solids via deep learning." *Nature Communications* **12**, 1506 (2021). DOI: 10.1038/s41467-021-21806-z.
    - A rotation-variant structural representation fed to a CNN predicts orientation-dependent shear-transformation propensity in 2D and 3D glasses. It transfers across processing histories and compositions. (Training-set size not checked **[unconfirmed]**.)

20. **Jung, G., Biroli, G., Berthier, L.** "Predicting dynamic heterogeneity in glass-forming liquids by physics-inspired machine learning." *Phys. Rev. Lett.* **130**, 238202 (2023). DOI: 10.1103/PhysRevLett.130.238202. arXiv:2210.16623.
    - GlassMLP is an MLP over coarse-grained physical descriptors, including coarse-grained **potential energy** and its local variance, with about 650 parameters. It beats GNNs while using about 10× less training data: 300 structures of N = 1290 (2D KA), split three ways, so about 100 are used for training. It transfers to larger systems.

21. **Boattini, E., Smallenburg, F., Filion, L.** "Averaging local structure to predict the dynamic propensity in supercooled liquids." *Phys. Rev. Lett.* **127**, 088007 (2021). DOI: 10.1103/PhysRevLett.127.088007.
    - Simple linear regression on neighbour-averaged structural descriptors matches GNN accuracy with far fewer parameters. The key ingredient is averaging over the neighbourhood, not model capacity.

22. **Shiba, H., Hanai, M., Suzumura, T., Shimokawabe, T.** "BOTAN: BOnd TArgeting Network for prediction of slow glassy dynamics by machine learning relative motion." *J. Chem. Phys.* **158**, 084503 (2023). DOI: 10.1063/5.0129791. arXiv:2206.14024.
    - A GNN with an edge (bond) decoder learns relative motion of neighbouring pairs and improves long-time propensity prediction. (One JHPCN report cites the article number as 184501; the arXiv journal-ref says 084503, which I used.)

23. **Font-Clos, F., Zanchi, M., Hiemer, S., Bonfanti, S., Guerra, R., Zaiser, M., Zapperi, S.** "Predicting the failure of two-dimensional silica glasses." *Nature Communications* **13**, 2820 (2022). DOI: 10.1038/s41467-022-30530-1.
    - A ResNet trained on 1000 configurations per dataset (3456 atoms each; 72/10/20 split, with augmentation) predicts sample failure strain. Its Grad-CAM attention maps **correlate strikingly with per-atom potential energy**, most strongly for samples that fail at low strain.

24. **Pezzicoli, F. S., Charpiat, G., Landes, F. P.** "Rotation-equivariant graph neural networks for learning glassy liquids representations." *SciPost Phys.* **16**, 136 (2024). DOI: 10.21468/SciPostPhys.16.5.136.
    - An SE(3)-equivariant GNN beats invariant GNNs at equal or smaller parameter count, generalises to temperatures it was not trained on, and supports transfer learning. This is the most direct precedent for an equivariant LYS surrogate.

25. **Wang, Q., Jain, A.** "A transferable machine-learning framework linking interstice distribution and plastic heterogeneity in metallic glasses." *Nature Communications* **10**, 5537 (2019) (article number **[unconfirmed]**). DOI: 10.1038/s41467-019-13511-9.
    - Interstice-distribution features plus ML ("quench-in softness") identify plastic sites in Cu–Zr. The model transfers across compositions and other metallic-glass systems.

26. **Xiao, H., Zhang, G., Yang, E., Ivancic, R., Ridout, S., Riggleman, R., Durian, D. J., Liu, A. J.** "Identifying microscopic factors that influence ductility in disordered solids." *PNAS* **120**(42), e2307552120 (2023). DOI: 10.1073/pnas.2307552120 (DOI inferred from article number). arXiv:2303.12486 (titled "Machine learning-informed structuro-elastoplasticity predicts ductility of disordered solids").
    - A ResNet image model predicts **per-particle local yield stress** from structure, and the prediction feeds a structuro-elastoplastic model of ductility. This is the closest prior LYS surrogate. Its training size and accuracy are in the SI, which I did not check **[unconfirmed]**.

27. **Jung, G., Alkemade, R. M., Bapst, V., Coslovich, D., Filion, L., Landes, F. P., Liu, A., Pezzicoli, F. S., Shiba, H., Volpe, G., Zamponi, F., Berthier, L., Biroli, G., et al.** "Roadmap on machine learning glassy dynamics." *Nature Reviews Physics* **7**, 91–104 (2025). DOI: 10.1038/s42254-024-00791-4. arXiv:2311.14752. (HAL lists the published title as "...glassy liquids"; full author list not checked.)
    - A community review that introduces the GlassBench benchmark datasets (2D and 3D). It is a useful reference for comparing data efficiency.

## C. Cheaper and faster AQS, minimisation and saddle search

28. **Maloney, C. E., Lemaître, A.** "Amorphous systems in athermal, quasistatic shear." *Phys. Rev. E* **74**, 016118 (2006). DOI: 10.1103/PhysRevE.74.016118. arXiv:cond-mat/0510677. (Volume and page confirmed only from a lecture slide that cites the paper.)
    - This is the standard AQS protocol: elastic branches interrupted by plastic events, which organise into slip lines.

29. **Maloney, C. E., Lacks, D. J.** "Energy barrier scalings in driven systems." *Phys. Rev. E* **73**, 061106 (2006). DOI: 10.1103/PhysRevE.73.061106.
    - Barriers vanish as ΔU ∝ δ^{3/2} near a loss of stability (catastrophe theory, i.e. saddle-node). This scaling is what lets us extrapolate a soft-mode eigenvalue to the instability strain.

30. **Bitzek, E., Koskinen, P., Gähler, F., Moseler, M., Gumbsch, P.** "Structural relaxation made simple." *Phys. Rev. Lett.* **97**, 170201 (2006). DOI: 10.1103/PhysRevLett.97.170201.
    - FIRE is MD with velocity mixing and an adaptive timestep. It is faster than CG and is the usual AQS minimiser.

31. **Guénolé, J., Nöhring, W. G., Vaid, A., Houllé, F., Xie, Z., Prakash, A., Bitzek, E.** "Assessment and optimization of the fast inertial relaxation engine (FIRE) for energy minimization in atomistic simulations and its implementation in LAMMPS." *Comput. Mater. Sci.* **175**, 109584 (2020). DOI: 10.1016/j.commatsci.2020.109584. (Full author list taken from arXiv:1908.02038 and not re-checked.)
    - FIRE 2.0: the choice of integrator and parameters strongly affects cost.

32. **Barkema, G. T., Mousseau, N.** "Event-based relaxation of continuous disordered systems." *Phys. Rev. Lett.* **77**, 4358–4361 (1996). DOI: 10.1103/PhysRevLett.77.4358.
    - This introduced the activation–relaxation technique (ART), which samples the energy landscape event by event (activation to a saddle, then relaxation).

33. **Malek, R., Mousseau, N.** "Dynamics of Lennard-Jones clusters: A characterization of the activation-relaxation technique." *Phys. Rev. E* **62**, 7723–7728 (2000). DOI: 10.1103/PhysRevE.62.7723.
    - ART nouveau, using Lanczos lowest-eigenvector activation, finds first-order saddles from a given minimum reversibly.

34. **Cancès, E., Legoll, F., Marinica, M.-C., Minoukadeh, K., Willaime, F.** "Some improvements of the activation-relaxation technique method for finding transition pathways on potential energy surfaces." *J. Chem. Phys.* **130**, 114711 (2009). DOI: 10.1063/1.3088532 **[DOI unconfirmed]**.
    - A provably convergent ARTn variant with faster convergence near the saddle.

35. **Henkelman, G., Jónsson, H.** "A dimer method for finding saddle points on high dimensional potential surfaces using only first derivatives." *J. Chem. Phys.* **111**, 7010 (1999). DOI: 10.1063/1.480097.
    - Single-ended saddle search using only forces, which scales to large systems.

36. **Henkelman, G., Uberuaga, B. P., Jónsson, H.** "A climbing image nudged elastic band method for finding saddle points and minimum energy paths." *J. Chem. Phys.* **113**, 9901 (2000). DOI: 10.1063/1.1329672.
    - CI-NEB is the standard double-ended method for minimum-energy paths and barriers.

37. **Schoenholz, S. S., Cubuk, E. D.** "JAX MD: A framework for differentiable physics." *NeurIPS* **33** (2020); journal version *J. Stat. Mech.* (2021) 124016, DOI: 10.1088/1742-5468/ac3ae9 (article number **[unconfirmed]**). arXiv:1912.04232.
    - Differentiable GPU MD and minimisation (including FIRE) that scales to about 10⁵ particles on one GPU. Many AQS minimisations can run batched on a GPU.

38. **Thompson, A. P., Aktulga, H. M., Berger, R., et al.** "LAMMPS – a flexible simulation tool for particle-based materials modeling at the atomic, meso, and continuum scales." *Comput. Phys. Commun.* **271**, 108171 (2022). DOI: 10.1016/j.cpc.2021.108171.
    - The canonical LAMMPS reference; LAMMPS includes FIRE 2.0 and is the reference engine for AQS and LYS.

39. **Anderson, J. A., Glaser, J., Glotzer, S. C.** "HOOMD-blue: A Python package for high-performance molecular dynamics and hard particle Monte Carlo simulations." *Comput. Mater. Sci.* **173**, 109363 (2020). DOI: 10.1016/j.commatsci.2019.109363.
    - A GPU-native MD engine with a Python API and FIRE minimisation.

*Adaptive and event-driven AQS:* I found **no dedicated peer-reviewed paper** on adaptive strain stepping for AQS. Richard et al. (2020) still use a fixed Δγ = 10⁻⁵ "in order not to miss plastic events." Xu et al. (2018, entry 11) and Maloney & Lacks (2006, entry 29) provide the theory for jumping straight to the next instability. Our 18× speed-up therefore appears to be a methodological contribution in its own right.

## D. Mesoscale elasto-plastic models that consume local yield thresholds

40. **Nicolas, A., Ferrero, E. E., Martens, K., Barrat, J.-L.** "Deformation and flow of amorphous solids: Insights from elastoplastic models." *Rev. Mod. Phys.* **90**, 045006 (2018). DOI: 10.1103/RevModPhys.90.045006.
    - The standard review of EPMs: mesoscale blocks yield at local thresholds and redistribute stress through the Eshelby propagator.

41. **Fernández Castellanos, D., Roux, S., Patinet, S.** "Insights from the quantitative calibration of an elasto-plastic model from a Lennard-Jones atomic glass." *Comptes Rendus Physique* **22**(S3), 135–162 (2021). DOI: 10.5802/crphys.48.
    - The EPM is calibrated on 2D LJ glasses, with local thresholds taken from atomistic LYS. Local agreement holds only for coarse probing scales and a subset of parameters.

42. **Castellanos, D. F., Roux, S., Patinet, S.** "History dependent plasticity of glass: A mapping between atomistic and elasto-plastic models." *Acta Materialia* **241**, 118405 (2022). DOI: 10.1016/j.actamat.2022.118405 **[DOI inferred]**.
    - LYS-informed, state-dependent EPM thresholds reproduce the transient, preparation-dependent response of quenched 2D glasses. This is the main downstream consumer for a fast LYS surrogate.

43. **Liu, C., Ferrero, E. E., Puosi, F., Barrat, J.-L., Martens, K.** "Driving rate dependence of avalanche statistics and shapes at the yielding transition." *Phys. Rev. Lett.* **116**, 065501 (2016). DOI: 10.1103/PhysRevLett.116.065501.
    - Particle simulations and an EPM agree on how avalanche statistics and shapes depend on driving rate, with non-mean-field exponents.

## E. Equivariant networks, active learning and data efficiency

44. **Batzner, S., Musaelian, A., Sun, L., Geiger, M., Mailoa, J. P., Kornbluth, M., Molinari, N., Smidt, T. E., Kozinsky, B.** "E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials." *Nature Communications* **13**, 2453 (2022). DOI: 10.1038/s41467-022-29939-5.
    - NequIP reaches state-of-the-art accuracy with up to 1000× less training data than invariant models. This is the main argument that equivariance could move our GNN's data threshold (about 8 glasses).

45. **Musaelian, A., Batzner, S., Johansson, A., Sun, L., Owen, C. J., Kornbluth, M., Kozinsky, B.** "Learning local equivariant representations for large-scale atomistic dynamics." *Nature Communications* **14**, 579 (2023). DOI: 10.1038/s41467-023-36329-y.
    - Allegro is a strictly local equivariant model with no message passing, scaling to 10⁸ atoms. Locality matches LYS, which is itself a local probe.

46. **Batatia, I., Kovács, D. P., Simm, G. N. C., Ortner, C., Csányi, G.** "MACE: Higher order equivariant message passing neural networks for fast and accurate force fields." *NeurIPS* **35** (2022). arXiv:2206.07697. (NeurIPS proceedings carry no DOI.)
    - Four-body equivariant messages mean only two message-passing layers are needed, which keeps the model fast.

47. **Smith, J. S., Nebgen, B., Lubbers, N., Isayev, O., Roitberg, A. E.** "Less is more: Sampling chemical space with active learning." *J. Chem. Phys.* **148**, 241733 (2018). DOI: 10.1063/1.5023802.
    - Query-by-committee active learning, using ensemble disagreement, reaches the same accuracy with a fraction of the data that random sampling needs. This is a template for choosing which atoms to label with expensive exact LYS.

48. **Lookman, T., Balachandran, P. V., Xue, D., Yuan, R.** "Active learning in materials science with emphasis on adaptive sampling using uncertainties for targeted design." *npj Computational Materials* **5**, 21 (2019). DOI: 10.1038/s41524-019-0153-8.
    - A review of surrogate-plus-uncertainty acquisition functions (EI, UCB, etc.) for choosing the next costly calculation.

---

## What the literature says about our "per-atom energy predicts events" finding

- **Richard et al. (2020)** included the local potential energy ϕ only as a control indicator "not a priori related to plasticity." They report that most purely structural indicators predict the first plastic event poorly, and that ϕ, like free volume, shows "very weak contrast" inside versus outside a future shear band until the band has formed. ϕ's exact C_min value appears only in their Fig. 4, which I could not read numerically.
- **Widmer-Cooper & Harrowell (2006)** likewise found inherent-structure energy per particle correlates poorly with propensity.
- **Against that**, Font-Clos et al. (2022) found CNN attention maps for failure tracking per-atom potential energy closely, and GlassMLP (Jung et al. 2023) uses coarse-grained energy as a key input.
- **Taken together:** energy is a weak indicator of the *first* event (the measure Richard et al. use), but it does carry information about where plasticity accumulates over many events. That fits our result that energy predicts events while being orthogonal to LYS, which is a threshold-distance measure that is not sensitive to energy.
- **Caveat:** I found no paper that reports a direct Spearman correlation between ϕ and LYS.

## How many training samples prior ML work used

| Study | Training data |
|---|---|
| Richard et al. 2020 | 100 independent samples of N = 10⁴ (benchmark statistics) |
| Bapst et al. 2020 | About 400 training configurations of N = 4096 per state point, out of 800 (secondary source) |
| GlassMLP (Jung et al. 2023) | About 100 training structures of N = 1290; reports needing about 10× less data than GNNs |
| Font-Clos et al. 2022 | About 720 samples of 3456 atoms, augmented to about 4×10⁴ images |

Our GNN's data threshold of about 8 × 1024-atom glasses is therefore **one to two orders of magnitude below typical training sets**. A physics-feature model plateauing at Spearman ≈ 0.5 is consistent with Boattini et al. and GlassMLP, who show that simple descriptors capture most of the signal that GNNs capture.
