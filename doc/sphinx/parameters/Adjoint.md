(parameters:Adjoint)=
# Adjoint


## **Subsection:** Adjoint


::::{dropdown} __Parameter:__ {ref}`List of objectives<parameters:Adjoint/List_20of_20objectives>`
:name: parameters:Adjoint/List_20of_20objectives
**Default value:**

**Pattern:** [MultipleSelection composition field|dynamic topography|geoid|self-gravitating geoid|temperature field ]

**Documentation:** Comma separated list of adjoint objective functionals.

The following objective functionals are available:

&lsquo;composition field&rsquo;: Terminal composition-field misfit objective for the time-dependent temperature adjoint workflow.

&lsquo;dynamic topography&rsquo;: Objective functional for instantaneous dynamic-topography inversion. The objective evaluates one half of the squared surface dynamic topography misfit and assembles the corresponding top-boundary adjoint right hand side.

&lsquo;geoid&rsquo;: Coefficient-space non-self-gravitating geoid objective. It evaluates one half of the diagonally weighted spherical-harmonic coefficient misfit for mantle density and optional surface/CMB dynamic topography. Enabled topography branches share one physical Stokes adjoint; the mantle-density contribution is the direct transpose of the shared harmonic operator.

&lsquo;self-gravitating geoid&rsquo;: Coupled self-gravitating geoid objective with top/CMB CBF stress, density-moment, physical Stokes, and boundary-mean density matrix terms. It excludes gravity, geometry, and external-density matrix derivatives.

&lsquo;temperature field&rsquo;: Terminal temperature-field misfit objective for the time-dependent temperature adjoint workflow.
::::

::::{dropdown} __Parameter:__ {ref}`Solve mode<parameters:Adjoint/Solve_20mode>`
:name: parameters:Adjoint/Solve_20mode
**Default value:** kernel

**Pattern:** [Selection kernel|optimize ]

**Documentation:** Run the ordinary adjoint kernel or the opt-in reduced initial-temperature ROL optimization.
::::

(parameters:Adjoint/Composition_20field)=
## **Subsection:** Adjoint / Composition field
::::{dropdown} __Parameter:__ {ref}`Coordinate system<parameters:Adjoint/Composition_20field/Coordinate_20system>`
:name: parameters:Adjoint/Composition_20field/Coordinate_20system
**Default value:** cartesian

**Pattern:** [Selection cartesian|spherical|depth ]

**Documentation:** Coordinate system used to evaluate the terminal target and weight functions.
::::

::::{dropdown} __Parameter:__ {ref}`Field name<parameters:Adjoint/Composition_20field/Field_20name>`
:name: parameters:Adjoint/Composition_20field/Field_20name
**Default value:**

**Pattern:** [Anything]

**Documentation:** Name of the compositional field used by the terminal composition misfit objective.
::::

::::{dropdown} __Parameter:__ {ref}`Observation source<parameters:Adjoint/Composition_20field/Observation_20source>`
:name: parameters:Adjoint/Composition_20field/Observation_20source
**Default value:** function

**Pattern:** [Selection function|file ]

**Documentation:** Reference-data source for this terminal scalar-field misfit. A file is read once and interpolated into the current ASPECT finite-element space.
::::

(parameters:Adjoint/Composition_20field/Observation_20file)=
## **Subsection:** Adjoint / Composition field / Observation file
::::{dropdown} __Parameter:__ {ref}`Data directory<parameters:Adjoint/Composition_20field/Observation_20file/Data_20directory>`
:name: parameters:Adjoint/Composition_20field/Observation_20file/Data_20directory
**Default value:** .

**Pattern:** [DirectoryName]

**Documentation:** The name of a directory that contains the model data. This path may either be absolute (if starting with a &lsquo;/&rsquo;) or relative to the current directory. The path may also include the special text &lsquo;$ASPECT_SOURCE_DIR&rsquo; which will be interpreted as the path in which the ASPECT source files were located when ASPECT was compiled. This interpretation allows, for example, to reference files located in the &lsquo;data/&rsquo; subdirectory of ASPECT. A trailing slash at the end of the directory path is optional; the plugin will automatically append a &rsquo;/&rsquo; when the parameters are parsed if it is missing.
::::

::::{dropdown} __Parameter:__ {ref}`Data file name<parameters:Adjoint/Composition_20field/Observation_20file/Data_20file_20name>`
:name: parameters:Adjoint/Composition_20field/Observation_20file/Data_20file_20name
**Default value:** observation.txt

**Pattern:** [Anything]

**Documentation:** The file name of the model data.
::::

::::{dropdown} __Parameter:__ {ref}`First point on slice<parameters:Adjoint/Composition_20field/Observation_20file/First_20point_20on_20slice>`
:name: parameters:Adjoint/Composition_20field/Observation_20file/First_20point_20on_20slice
**Default value:** 0.0,1.0,0.0

**Pattern:** [Anything]

**Documentation:** Point that determines the plane in which the 2d slice lies in. This variable is only used if &rsquo;Slice dataset in 2d plane&rsquo; is true. The slice will go through this point, the point defined by the parameter &rsquo;Second point on slice&rsquo;, and the center of the model domain. After the rotation, this first point will lie along the (0,1,0) axis of the coordinate system. The coordinates of the point have to be given in Cartesian coordinates.
::::

::::{dropdown} __Parameter:__ {ref}`Scale factor<parameters:Adjoint/Composition_20field/Observation_20file/Scale_20factor>`
:name: parameters:Adjoint/Composition_20field/Observation_20file/Scale_20factor
**Default value:** 1.

**Pattern:** [Double -MAX_DOUBLE...MAX_DOUBLE (inclusive)]

**Documentation:** Scalar factor, which is applied to the model data. You might want to use this to scale the input to a reference model. Another way to use this factor is to convert units of the input files. For instance, if you provide velocities in cm/yr set this factor to 0.01.
::::

::::{dropdown} __Parameter:__ {ref}`Second point on slice<parameters:Adjoint/Composition_20field/Observation_20file/Second_20point_20on_20slice>`
:name: parameters:Adjoint/Composition_20field/Observation_20file/Second_20point_20on_20slice
**Default value:** 1.0,0.0,0.0

**Pattern:** [Anything]

**Documentation:** Second point that determines the plane in which the 2d slice lies in. This variable is only used if &rsquo;Slice dataset in 2d plane&rsquo; is true. The slice will go through this point, the point defined by the parameter &rsquo;First point on slice&rsquo;, and the center of the model domain. The coordinates of the point have to be given in Cartesian coordinates.
::::

::::{dropdown} __Parameter:__ {ref}`Slice dataset in 2D plane<parameters:Adjoint/Composition_20field/Observation_20file/Slice_20dataset_20in_202D_20plane>`
:name: parameters:Adjoint/Composition_20field/Observation_20file/Slice_20dataset_20in_202D_20plane
**Default value:** false

**Pattern:** [Bool]

**Documentation:** Whether to use a 2d data slice of a 3d data file or the entire data file. Slicing a 3d dataset is only supported for 2d models.
::::

(parameters:Adjoint/Composition_20field/Target_20function)=
## **Subsection:** Adjoint / Composition field / Target function
::::{dropdown} __Parameter:__ {ref}`Function constants<parameters:Adjoint/Composition_20field/Target_20function/Function_20constants>`
:name: parameters:Adjoint/Composition_20field/Target_20function/Function_20constants
**Default value:**

**Pattern:** [Anything]

**Documentation:** Sometimes it is convenient to use symbolic constants in the expression that describes the function, rather than having to use its numeric value everywhere the constant appears. These values can be defined using this parameter, in the form &lsquo;var1=value1, var2=value2, ...&rsquo;.

A typical example would be to set this runtime parameter to &lsquo;pi=3.1415926536&rsquo; and then use &lsquo;pi&rsquo; in the expression of the actual formula. (That said, for convenience this class actually defines both &lsquo;pi&rsquo; and &lsquo;Pi&rsquo; by default, but you get the idea.)
::::

::::{dropdown} __Parameter:__ {ref}`Function expression<parameters:Adjoint/Composition_20field/Target_20function/Function_20expression>`
:name: parameters:Adjoint/Composition_20field/Target_20function/Function_20expression
**Default value:** 0

**Pattern:** [Anything]

**Documentation:** Target final scalar field.
::::

::::{dropdown} __Parameter:__ {ref}`Variable names<parameters:Adjoint/Composition_20field/Target_20function/Variable_20names>`
:name: parameters:Adjoint/Composition_20field/Target_20function/Variable_20names
**Default value:** x,y,t

**Pattern:** [Anything]

**Documentation:** The names of the variables as they will be used in the function, separated by commas. By default, the names of variables at which the function will be evaluated are &lsquo;x&rsquo; (in 1d), &lsquo;x,y&rsquo; (in 2d) or &lsquo;x,y,z&rsquo; (in 3d) for spatial coordinates and &lsquo;t&rsquo; for time. You can then use these variable names in your function expression and they will be replaced by the values of these variables at which the function is currently evaluated. However, you can also choose a different set of names for the independent variables at which to evaluate your function expression. For example, if you work in spherical coordinates, you may wish to set this input parameter to &lsquo;r,phi,theta,t&rsquo; and then use these variable names in your function expression.
::::

(parameters:Adjoint/Composition_20field/Weight_20function)=
## **Subsection:** Adjoint / Composition field / Weight function
::::{dropdown} __Parameter:__ {ref}`Function constants<parameters:Adjoint/Composition_20field/Weight_20function/Function_20constants>`
:name: parameters:Adjoint/Composition_20field/Weight_20function/Function_20constants
**Default value:**

**Pattern:** [Anything]

**Documentation:** Sometimes it is convenient to use symbolic constants in the expression that describes the function, rather than having to use its numeric value everywhere the constant appears. These values can be defined using this parameter, in the form &lsquo;var1=value1, var2=value2, ...&rsquo;.

A typical example would be to set this runtime parameter to &lsquo;pi=3.1415926536&rsquo; and then use &lsquo;pi&rsquo; in the expression of the actual formula. (That said, for convenience this class actually defines both &lsquo;pi&rsquo; and &lsquo;Pi&rsquo; by default, but you get the idea.)
::::

::::{dropdown} __Parameter:__ {ref}`Function expression<parameters:Adjoint/Composition_20field/Weight_20function/Function_20expression>`
:name: parameters:Adjoint/Composition_20field/Weight_20function/Function_20expression
**Default value:** 1

**Pattern:** [Anything]

**Documentation:** Non-negative spatial weight for the terminal scalar-field misfit.
::::

::::{dropdown} __Parameter:__ {ref}`Variable names<parameters:Adjoint/Composition_20field/Weight_20function/Variable_20names>`
:name: parameters:Adjoint/Composition_20field/Weight_20function/Variable_20names
**Default value:** x,y,t

**Pattern:** [Anything]

**Documentation:** The names of the variables as they will be used in the function, separated by commas. By default, the names of variables at which the function will be evaluated are &lsquo;x&rsquo; (in 1d), &lsquo;x,y&rsquo; (in 2d) or &lsquo;x,y,z&rsquo; (in 3d) for spatial coordinates and &lsquo;t&rsquo; for time. You can then use these variable names in your function expression and they will be replaced by the values of these variables at which the function is currently evaluated. However, you can also choose a different set of names for the independent variables at which to evaluate your function expression. For example, if you work in spherical coordinates, you may wish to set this input parameter to &lsquo;r,phi,theta,t&rsquo; and then use these variable names in your function expression.
::::

(parameters:Adjoint/Diagnostics)=
## **Subsection:** Adjoint / Diagnostics
::::{dropdown} __Parameter:__ {ref}`Artificial viscosity coefficient seed test epsilon<parameters:Adjoint/Diagnostics/Artificial_20viscosity_20coefficient_20seed_20test_20epsilon>`
:name: parameters:Adjoint/Diagnostics/Artificial_20viscosity_20coefficient_20seed_20test_20epsilon
**Default value:** 1e-7

**Pattern:** [Double 0...MAX_DOUBLE (inclusive)]

**Documentation:** Finite-difference epsilon used by the artificial viscosity coefficient seed diagnostic.
::::

::::{dropdown} __Parameter:__ {ref}`Artificial viscosity estimator transpose test epsilon<parameters:Adjoint/Diagnostics/Artificial_20viscosity_20estimator_20transpose_20test_20epsilon>`
:name: parameters:Adjoint/Diagnostics/Artificial_20viscosity_20estimator_20transpose_20test_20epsilon
**Default value:** 1e-7

**Pattern:** [Double 0...MAX_DOUBLE (inclusive)]

**Documentation:** Finite-difference epsilon used by the artificial viscosity estimator transpose diagnostic.
::::

::::{dropdown} __Parameter:__ {ref}`C to Stokes RHS transpose test epsilon<parameters:Adjoint/Diagnostics/C_20to_20Stokes_20RHS_20transpose_20test_20epsilon>`
:name: parameters:Adjoint/Diagnostics/C_20to_20Stokes_20RHS_20transpose_20test_20epsilon
**Default value:** 1e-7

**Pattern:** [Double 0...MAX_DOUBLE (inclusive)]

**Documentation:** Finite-difference epsilon used by the C to Stokes RHS transpose diagnostic.
::::

::::{dropdown} __Parameter:__ {ref}`Checkpoint transaction failure injection<parameters:Adjoint/Diagnostics/Checkpoint_20transaction_20failure_20injection>`
:name: parameters:Adjoint/Diagnostics/Checkpoint_20transaction_20failure_20injection
**Default value:** none

**Pattern:** [Selection none|after temporary checkpoint directory creation|after rank-local checkpoint payload writing|after all payloads before metadata publication|after checkpoint metadata publication|after committed checkpoint directory before latest publication|after latest publication ]

**Documentation:** Benchmark-only MPI checkpoint transaction failure point. The selected rank reports a controlled failure at this collective boundary; all ranks then exit through the same quiet path.
::::

::::{dropdown} __Parameter:__ {ref}`Checkpoint transaction failure next reverse step<parameters:Adjoint/Diagnostics/Checkpoint_20transaction_20failure_20next_20reverse_20step>`
:name: parameters:Adjoint/Diagnostics/Checkpoint_20transaction_20failure_20next_20reverse_20step
**Default value:** -1

**Pattern:** [Integer range -1...2147483647 (inclusive)]

**Documentation:** Reverse checkpoint frontier at which to inject the benchmark-only transaction failure. A negative value disables the injection.
::::

::::{dropdown} __Parameter:__ {ref}`Checkpoint transaction failure rank<parameters:Adjoint/Diagnostics/Checkpoint_20transaction_20failure_20rank>`
:name: parameters:Adjoint/Diagnostics/Checkpoint_20transaction_20failure_20rank
**Default value:** -1

**Pattern:** [Integer range -1...2147483647 (inclusive)]

**Documentation:** MPI rank used for the benchmark-only checkpoint failure injection. -1 selects rank zero.
::::

::::{dropdown} __Parameter:__ {ref}`Direct Stokes viscosity derivative test epsilon<parameters:Adjoint/Diagnostics/Direct_20Stokes_20viscosity_20derivative_20test_20epsilon>`
:name: parameters:Adjoint/Diagnostics/Direct_20Stokes_20viscosity_20derivative_20test_20epsilon
**Default value:** 1e-7

**Pattern:** [Double 0...MAX_DOUBLE (inclusive)]

**Documentation:** Finite-difference epsilon used by the direct Stokes viscosity derivative diagnostic.
::::

::::{dropdown} __Parameter:__ {ref}`Global entropy denominator scalar test epsilon<parameters:Adjoint/Diagnostics/Global_20entropy_20denominator_20scalar_20test_20epsilon>`
:name: parameters:Adjoint/Diagnostics/Global_20entropy_20denominator_20scalar_20test_20epsilon
**Default value:** 1e-7

**Pattern:** [Double 0...MAX_DOUBLE (inclusive)]

**Documentation:** Finite-difference epsilon used by the global entropy denominator scalar diagnostic.
::::

::::{dropdown} __Parameter:__ {ref}`Global entropy variation transpose test epsilon<parameters:Adjoint/Diagnostics/Global_20entropy_20variation_20transpose_20test_20epsilon>`
:name: parameters:Adjoint/Diagnostics/Global_20entropy_20variation_20transpose_20test_20epsilon
**Default value:** 1e-7

**Pattern:** [Double 0...MAX_DOUBLE (inclusive)]

**Documentation:** Finite-difference epsilon used by the global entropy variation transpose diagnostic.
::::

::::{dropdown} __Parameter:__ {ref}`Lifecycle failure injection<parameters:Adjoint/Diagnostics/Lifecycle_20failure_20injection>`
:name: parameters:Adjoint/Diagnostics/Lifecycle_20failure_20injection
**Default value:** none

**Pattern:** [Selection none|before checkpoint commit|after checkpoint commit before cleanup|during cleanup|after cleanup ]

**Documentation:** R0 test hook that deliberately interrupts the reverse lifecycle at a transaction boundary. Do not enable this in a production run.
::::

::::{dropdown} __Parameter:__ {ref}`Local entropy residual transpose test epsilon<parameters:Adjoint/Diagnostics/Local_20entropy_20residual_20transpose_20test_20epsilon>`
:name: parameters:Adjoint/Diagnostics/Local_20entropy_20residual_20transpose_20test_20epsilon
**Default value:** 1e-7

**Pattern:** [Double 0...MAX_DOUBLE (inclusive)]

**Documentation:** Finite-difference epsilon used by the local entropy residual transpose diagnostic.
::::

::::{dropdown} __Parameter:__ {ref}`Material model adjoint VJP test epsilon<parameters:Adjoint/Diagnostics/Material_20model_20adjoint_20VJP_20test_20epsilon>`
:name: parameters:Adjoint/Diagnostics/Material_20model_20adjoint_20VJP_20test_20epsilon
**Default value:** 1e-7

**Pattern:** [Double 0...MAX_DOUBLE (inclusive)]

**Documentation:** Finite-difference epsilon used by the standalone MaterialModelAdjoint VJP diagnostic.
::::

::::{dropdown} __Parameter:__ {ref}`Output MPI adjoint memory diagnostics<parameters:Adjoint/Diagnostics/Output_20MPI_20adjoint_20memory_20diagnostics>`
:name: parameters:Adjoint/Diagnostics/Output_20MPI_20adjoint_20memory_20diagnostics
**Default value:** false

**Pattern:** [Bool]

**Documentation:** Emit a rank-local RSS and distributed-vector checkpoint around replay reconstruction and the first reverse Stokes setup. This is an allocation-free diagnostic for the experimental MPI reverse path.
::::

::::{dropdown} __Parameter:__ {ref}`Output terminal seed diagnostics<parameters:Adjoint/Diagnostics/Output_20terminal_20seed_20diagnostics>`
:name: parameters:Adjoint/Diagnostics/Output_20terminal_20seed_20diagnostics
**Default value:** false

**Pattern:** [Bool]

**Documentation:** Write rank-owned entries and global block norms of the terminal adjoint seeds after their assembly. This is a diagnostic and does not alter the reverse solve.
::::

::::{dropdown} __Parameter:__ {ref}`Run C to Stokes RHS transpose test<parameters:Adjoint/Diagnostics/Run_20C_20to_20Stokes_20RHS_20transpose_20test>`
:name: parameters:Adjoint/Diagnostics/Run_20C_20to_20Stokes_20RHS_20transpose_20test
**Default value:** false

**Pattern:** [Bool]

**Documentation:** Run a finite-difference dot-product diagnostic for the composition dependence of the assembled Stokes right hand side. This is intended for small adjoint benchmarks only.
::::

::::{dropdown} __Parameter:__ {ref}`Run advection preconditioner comparison<parameters:Adjoint/Diagnostics/Run_20advection_20preconditioner_20comparison>`
:name: parameters:Adjoint/Diagnostics/Run_20advection_20preconditioner_20comparison
**Default value:** false

**Pattern:** [Bool]

**Documentation:** For the first temperature and composition transpose solves, compare unpreconditioned GMRES with ILU(A^T) using the same restored matrix and right hand side. This diagnostic does not alter the production adjoint solve.
::::

::::{dropdown} __Parameter:__ {ref}`Run artificial viscosity coefficient seed test<parameters:Adjoint/Diagnostics/Run_20artificial_20viscosity_20coefficient_20seed_20test>`
:name: parameters:Adjoint/Diagnostics/Run_20artificial_20viscosity_20coefficient_20seed_20test
**Default value:** false

**Pattern:** [Bool]

**Documentation:** Run an isolated finite-difference dot-product diagnostic for the derivative of the advection residual with respect to independent cellwise artificial-viscosity coefficients.
::::

::::{dropdown} __Parameter:__ {ref}`Run artificial viscosity estimator transpose test<parameters:Adjoint/Diagnostics/Run_20artificial_20viscosity_20estimator_20transpose_20test>`
:name: parameters:Adjoint/Diagnostics/Run_20artificial_20viscosity_20estimator_20transpose_20test
**Default value:** false

**Pattern:** [Bool]

**Documentation:** Run a standalone finite-difference dot-product diagnostic for the transpose of the entropy-viscosity estimator from cellwise artificial-viscosity seed to old/old-old field and velocity states.
::::

::::{dropdown} __Parameter:__ {ref}`Run direct Stokes viscosity derivative test<parameters:Adjoint/Diagnostics/Run_20direct_20Stokes_20viscosity_20derivative_20test>`
:name: parameters:Adjoint/Diagnostics/Run_20direct_20Stokes_20viscosity_20derivative_20test
**Default value:** false

**Pattern:** [Bool]

**Documentation:** Run a direct finite-difference dot-product diagnostic for the Stokes residual derivative with respect to independent quadrature-point viscosity values.
::::

::::{dropdown} __Parameter:__ {ref}`Run global entropy denominator scalar test<parameters:Adjoint/Diagnostics/Run_20global_20entropy_20denominator_20scalar_20test>`
:name: parameters:Adjoint/Diagnostics/Run_20global_20entropy_20denominator_20scalar_20test
**Default value:** false

**Pattern:** [Bool]

**Documentation:** Run a scalar finite-difference diagnostic for the global_entropy_variation denominator in the alpha=2 entropy-viscosity estimator.
::::

::::{dropdown} __Parameter:__ {ref}`Run global entropy variation transpose test<parameters:Adjoint/Diagnostics/Run_20global_20entropy_20variation_20transpose_20test>`
:name: parameters:Adjoint/Diagnostics/Run_20global_20entropy_20variation_20transpose_20test
**Default value:** false

**Pattern:** [Bool]

**Documentation:** Run a standalone finite-difference diagnostic for global_entropy_variation(field_history), freezing the average_field supplied by the extrapolated field range.
::::

::::{dropdown} __Parameter:__ {ref}`Run local entropy residual transpose test<parameters:Adjoint/Diagnostics/Run_20local_20entropy_20residual_20transpose_20test>`
:name: parameters:Adjoint/Diagnostics/Run_20local_20entropy_20residual_20transpose_20test
**Default value:** false

**Pattern:** [Bool]

**Documentation:** Run a frozen-denominator diagnostic for the local max-residual/entropy-deviation numerator of the alpha=2 entropy-viscosity estimator.
::::

::::{dropdown} __Parameter:__ {ref}`Run material model adjoint VJP test<parameters:Adjoint/Diagnostics/Run_20material_20model_20adjoint_20VJP_20test>`
:name: parameters:Adjoint/Diagnostics/Run_20material_20model_20adjoint_20VJP_20test
**Default value:** false

**Pattern:** [Bool]

**Documentation:** Run a standalone finite-difference dot-product diagnostic for the MaterialModelAdjoint VJP backend. This v1 diagnostic is restricted to the simple material model.
::::

::::{dropdown} __Parameter:__ {ref}`Run matrix-free cold rebuild diagnostic<parameters:Adjoint/Diagnostics/Run_20matrix_2dfree_20cold_20rebuild_20diagnostic>`
:name: parameters:Adjoint/Diagnostics/Run_20matrix_2dfree_20cold_20rebuild_20diagnostic
**Default value:** false

**Pattern:** [Bool]

**Documentation:** Immediately before the first reverse Stokes solve, discard the matrix-free/GMG runtime state and let the ordinary adjoint-restart reconstruction path rebuild it. This is a targeted A/B diagnostic and does not alter adjoint algebra or checkpoint data.
::::

::::{dropdown} __Parameter:__ {ref}`Run sparse trajectory replay validation<parameters:Adjoint/Diagnostics/Run_20sparse_20trajectory_20replay_20validation>`
:name: parameters:Adjoint/Diagnostics/Run_20sparse_20trajectory_20replay_20validation
**Default value:** false

**Pattern:** [Bool]

**Documentation:** Retain a separate dense full-trajectory reference archive and compare sparse interval replay states against it. This reference is diagnostic-only and is not used to satisfy reverse dependencies.
::::

::::{dropdown} __Parameter:__ {ref}`Run trajectory Stokes recovery diagnostic<parameters:Adjoint/Diagnostics/Run_20trajectory_20Stokes_20recovery_20diagnostic>`
:name: parameters:Adjoint/Diagnostics/Run_20trajectory_20Stokes_20recovery_20diagnostic
**Default value:** false

**Pattern:** [Bool]

**Documentation:** For stokes cached trajectories, recover Stokes fields from archived temperature and composition at representative accepted steps and compare them with the archived forward velocity and pressure. This validates the minimal-mode reconstruction path without changing the adjoint equations.
::::

::::{dropdown} __Parameter:__ {ref}`Run trajectory replay diagnostic<parameters:Adjoint/Diagnostics/Run_20trajectory_20replay_20diagnostic>`
:name: parameters:Adjoint/Diagnostics/Run_20trajectory_20replay_20diagnostic
**Default value:** false

**Pattern:** [Bool]

**Documentation:** Phase-A validation hook: replay the committed full trajectory with its recorded accepted-step schedule and compare every produced state with the archived reference. This does not enable sparse anchors or change the reverse sweep.
::::

::::{dropdown} __Parameter:__ {ref}`Run transposed advection constraint test<parameters:Adjoint/Diagnostics/Run_20transposed_20advection_20constraint_20test>`
:name: parameters:Adjoint/Diagnostics/Run_20transposed_20advection_20constraint_20test
**Default value:** false

**Pattern:** [Bool]

**Documentation:** Run a regression diagnostic that checks the transposed advection solve enforces homogeneous adjoint values on constrained advection DoFs.
::::

::::{dropdown} __Parameter:__ {ref}`Stop after adjoint checkpoint<parameters:Adjoint/Diagnostics/Stop_20after_20adjoint_20checkpoint>`
:name: parameters:Adjoint/Diagnostics/Stop_20after_20adjoint_20checkpoint
**Default value:** -1

**Pattern:** [Integer range -1...2147483647 (inclusive)]

**Documentation:** Testing hook: after committing a checkpoint whose next reverse step equals this value, terminate deliberately. A negative value disables the hook.
::::

::::{dropdown} __Parameter:__ {ref}`Trajectory Stokes recovery tolerance<parameters:Adjoint/Diagnostics/Trajectory_20Stokes_20recovery_20tolerance>`
:name: parameters:Adjoint/Diagnostics/Trajectory_20Stokes_20recovery_20tolerance
**Default value:** 1e-8

**Pattern:** [Double 0...MAX_DOUBLE (inclusive)]

**Documentation:** Maximum permitted relative velocity and pressure error in the trajectory Stokes recovery diagnostic after pressure-gauge normalization.
::::

::::{dropdown} __Parameter:__ {ref}`Trajectory append failure injection<parameters:Adjoint/Diagnostics/Trajectory_20append_20failure_20injection>`
:name: parameters:Adjoint/Diagnostics/Trajectory_20append_20failure_20injection
**Default value:** none

**Pattern:** [Selection none|after local serialization before shared-file write|after collective payload write before manifest|after payload write before manifest|after manifest publication ]

**Documentation:** Benchmark-only shared trajectory append transaction failure point. All ranks stop through QuietException; do not enable in production.
::::

::::{dropdown} __Parameter:__ {ref}`Trajectory append failure rank<parameters:Adjoint/Diagnostics/Trajectory_20append_20failure_20rank>`
:name: parameters:Adjoint/Diagnostics/Trajectory_20append_20failure_20rank
**Default value:** -1

**Pattern:** [Integer range -1...2147483647 (inclusive)]

**Documentation:** MPI rank used for the benchmark-only trajectory append failure injection. -1 selects rank zero.
::::

::::{dropdown} __Parameter:__ {ref}`Trajectory append failure step<parameters:Adjoint/Diagnostics/Trajectory_20append_20failure_20step>`
:name: parameters:Adjoint/Diagnostics/Trajectory_20append_20failure_20step
**Default value:** -1

**Pattern:** [Integer range -1...2147483647 (inclusive)]

**Documentation:** Forward trajectory record at which to inject the benchmark-only append failure. A negative value disables it.
::::

::::{dropdown} __Parameter:__ {ref}`Trajectory cleanup failure injection<parameters:Adjoint/Diagnostics/Trajectory_20cleanup_20failure_20injection>`
:name: parameters:Adjoint/Diagnostics/Trajectory_20cleanup_20failure_20injection
**Default value:** none

**Pattern:** [Selection none|before cleanup starts|after rank-local cleanup payload deletion|before cleanup collective success|after rank-local cleanup success before manifest update|after physical segment deletion before manifest update|after cleanup manifest update ]

**Documentation:** Benchmark-only MPI cleanup transaction failure point. The selected rank reports a controlled failure and all ranks stop together. Sparse cleanup can safely repeat an interrupted pre-manifest segment deletion from its durable checkpoint frontier.
::::

::::{dropdown} __Parameter:__ {ref}`Trajectory cleanup failure next reverse step<parameters:Adjoint/Diagnostics/Trajectory_20cleanup_20failure_20next_20reverse_20step>`
:name: parameters:Adjoint/Diagnostics/Trajectory_20cleanup_20failure_20next_20reverse_20step
**Default value:** -1

**Pattern:** [Integer range -1...2147483647 (inclusive)]

**Documentation:** Checkpoint frontier at which to inject the benchmark-only cleanup failure. A negative value disables injection.
::::

::::{dropdown} __Parameter:__ {ref}`Trajectory cleanup failure rank<parameters:Adjoint/Diagnostics/Trajectory_20cleanup_20failure_20rank>`
:name: parameters:Adjoint/Diagnostics/Trajectory_20cleanup_20failure_20rank
**Default value:** -1

**Pattern:** [Integer range -1...2147483647 (inclusive)]

**Documentation:** MPI rank used for the benchmark-only cleanup failure injection. -1 selects rank zero.
::::

::::{dropdown} __Parameter:__ {ref}`Trajectory replay tolerance<parameters:Adjoint/Diagnostics/Trajectory_20replay_20tolerance>`
:name: parameters:Adjoint/Diagnostics/Trajectory_20replay_20tolerance
**Default value:** 1e-11

**Pattern:** [Double 0...MAX_DOUBLE (inclusive)]

**Documentation:** Maximum relative state, velocity, and gauge-normalized pressure difference accepted by the Phase-A full-archive trajectory replay diagnostic.
::::

(parameters:Adjoint/Dynamic_20topography)=
## **Subsection:** Adjoint / Dynamic topography
::::{dropdown} __Parameter:__ {ref}`Target topography model<parameters:Adjoint/Dynamic_20topography/Target_20topography_20model>`
:name: parameters:Adjoint/Dynamic_20topography/Target_20topography_20model
**Default value:** none

**Pattern:** [Selection none|function ]

**Documentation:** The model used to prescribe the target dynamic topography. The dynamic topography objective evaluates one half of the squared misfit between the computed and target topography. A target topography model is required when this objective is used.
::::

(parameters:Adjoint/Dynamic_20topography/Function)=
## **Subsection:** Adjoint / Dynamic topography / Function
::::{dropdown} __Parameter:__ {ref}`Coordinate system<parameters:Adjoint/Dynamic_20topography/Function/Coordinate_20system>`
:name: parameters:Adjoint/Dynamic_20topography/Function/Coordinate_20system
**Default value:** spherical

**Pattern:** [Selection cartesian|spherical|depth ]

**Documentation:** A selection that determines the assumed coordinate system for the target function variables. Allowed values are &lsquo;cartesian&rsquo;, &lsquo;spherical&rsquo;, and &lsquo;depth&rsquo;. &lsquo;spherical&rsquo; coordinates are interpreted as r,phi or r,phi,theta in 2d/3d respectively with theta being the polar angle.
::::

::::{dropdown} __Parameter:__ {ref}`Function constants<parameters:Adjoint/Dynamic_20topography/Function/Function_20constants>`
:name: parameters:Adjoint/Dynamic_20topography/Function/Function_20constants
**Default value:**

**Pattern:** [Anything]

**Documentation:** Sometimes it is convenient to use symbolic constants in the expression that describes the function, rather than having to use its numeric value everywhere the constant appears. These values can be defined using this parameter, in the form &lsquo;var1=value1, var2=value2, ...&rsquo;.

A typical example would be to set this runtime parameter to &lsquo;pi=3.1415926536&rsquo; and then use &lsquo;pi&rsquo; in the expression of the actual formula. (That said, for convenience this class actually defines both &lsquo;pi&rsquo; and &lsquo;Pi&rsquo; by default, but you get the idea.)
::::

::::{dropdown} __Parameter:__ {ref}`Function expression<parameters:Adjoint/Dynamic_20topography/Function/Function_20expression>`
:name: parameters:Adjoint/Dynamic_20topography/Function/Function_20expression
**Default value:**

**Pattern:** [Anything]

**Documentation:** Target dynamic topography in meters.
::::

::::{dropdown} __Parameter:__ {ref}`Variable names<parameters:Adjoint/Dynamic_20topography/Function/Variable_20names>`
:name: parameters:Adjoint/Dynamic_20topography/Function/Variable_20names
**Default value:** r,phi

**Pattern:** [Anything]

**Documentation:** Names of the coordinates used in the target topography function.
::::

(parameters:Adjoint/Geoid)=
## **Subsection:** Adjoint / Geoid
::::{dropdown} __Parameter:__ {ref}`CMB density below<parameters:Adjoint/Geoid/CMB_20density_20below>`
:name: parameters:Adjoint/Geoid/CMB_20density_20below
**Default value:** 9900

**Pattern:** [Double 0...MAX_DOUBLE (inclusive)]

**Documentation:** Density below the CMB used by the CMB dynamic-topography and geoid operators.
::::

::::{dropdown} __Parameter:__ {ref}`Coefficient weights<parameters:Adjoint/Geoid/Coefficient_20weights>`
:name: parameters:Adjoint/Geoid/Coefficient_20weights
**Default value:**

**Pattern:** [Anything]

**Documentation:** Optional comma-separated nonnegative diagonal weights. One weight applies to both cosine and sine components of each degree/order; an empty list means all weights are one.
::::

::::{dropdown} __Parameter:__ {ref}`Include CMB dynamic topography contribution<parameters:Adjoint/Geoid/Include_20CMB_20dynamic_20topography_20contribution>`
:name: parameters:Adjoint/Geoid/Include_20CMB_20dynamic_20topography_20contribution
**Default value:** false

**Pattern:** [Bool]

**Documentation:** Include the CMB dynamic-topography geoid contribution through the bottom CBF transpose.
::::

::::{dropdown} __Parameter:__ {ref}`Include surface dynamic topography contribution<parameters:Adjoint/Geoid/Include_20surface_20dynamic_20topography_20contribution>`
:name: parameters:Adjoint/Geoid/Include_20surface_20dynamic_20topography_20contribution
**Default value:** false

**Pattern:** [Bool]

**Documentation:** Include the surface dynamic-topography geoid contribution through the validated top CBF transpose.
::::

::::{dropdown} __Parameter:__ {ref}`Maximum spherical harmonic degree<parameters:Adjoint/Geoid/Maximum_20spherical_20harmonic_20degree>`
:name: parameters:Adjoint/Geoid/Maximum_20spherical_20harmonic_20degree
**Default value:** 0

**Pattern:** [Integer range 0...2147483647 (inclusive)]

**Documentation:** Maximum degree l_max in the shared degree-major geoid coefficient layout.
::::

::::{dropdown} __Parameter:__ {ref}`Minimum spherical harmonic degree<parameters:Adjoint/Geoid/Minimum_20spherical_20harmonic_20degree>`
:name: parameters:Adjoint/Geoid/Minimum_20spherical_20harmonic_20degree
**Default value:** 0

**Pattern:** [Integer range 0...2147483647 (inclusive)]

**Documentation:** Minimum degree l_min in the shared degree-major geoid coefficient layout.
::::

::::{dropdown} __Parameter:__ {ref}`Observed cosine coefficients<parameters:Adjoint/Geoid/Observed_20cosine_20coefficients>`
:name: parameters:Adjoint/Geoid/Observed_20cosine_20coefficients
**Default value:**

**Pattern:** [Anything]

**Documentation:** Comma-separated observed cosine coefficients in shared degree-major order.
::::

::::{dropdown} __Parameter:__ {ref}`Observed sine coefficients<parameters:Adjoint/Geoid/Observed_20sine_20coefficients>`
:name: parameters:Adjoint/Geoid/Observed_20sine_20coefficients
**Default value:**

**Pattern:** [Anything]

**Documentation:** Comma-separated observed sine coefficients in shared degree-major order.
::::

::::{dropdown} __Parameter:__ {ref}`Surface density above<parameters:Adjoint/Geoid/Surface_20density_20above>`
:name: parameters:Adjoint/Geoid/Surface_20density_20above
**Default value:** 0

**Pattern:** [Double 0...MAX_DOUBLE (inclusive)]

**Documentation:** Density above the surface used by the dynamic-topography and surface-geoid operators.
::::

(parameters:Adjoint/Optimization)=
## **Subsection:** Adjoint / Optimization
::::{dropdown} __Parameter:__ {ref}`Gradient tolerance<parameters:Adjoint/Optimization/Gradient_20tolerance>`
:name: parameters:Adjoint/Optimization/Gradient_20tolerance
**Default value:** 1e-8

**Pattern:** [Double 0...MAX_DOUBLE (inclusive)]

**Documentation:**
::::

::::{dropdown} __Parameter:__ {ref}`Initial trust-region radius<parameters:Adjoint/Optimization/Initial_20trust_2dregion_20radius>`
:name: parameters:Adjoint/Optimization/Initial_20trust_2dregion_20radius
**Default value:** 1e-3

**Pattern:** [Double 0...MAX_DOUBLE (inclusive)]

**Documentation:**
::::

::::{dropdown} __Parameter:__ {ref}`L-BFGS history size<parameters:Adjoint/Optimization/L_2dBFGS_20history_20size>`
:name: parameters:Adjoint/Optimization/L_2dBFGS_20history_20size
**Default value:** 3

**Pattern:** [Integer range 1...2147483647 (inclusive)]

**Documentation:**
::::

::::{dropdown} __Parameter:__ {ref}`Lower bound<parameters:Adjoint/Optimization/Lower_20bound>`
:name: parameters:Adjoint/Optimization/Lower_20bound
**Default value:** -1e-2

**Pattern:** [Double -MAX_DOUBLE...MAX_DOUBLE (inclusive)]

**Documentation:**
::::

::::{dropdown} __Parameter:__ {ref}`Maximum iterations<parameters:Adjoint/Optimization/Maximum_20iterations>`
:name: parameters:Adjoint/Optimization/Maximum_20iterations
**Default value:** 1

**Pattern:** [Integer range 0...2147483647 (inclusive)]

**Documentation:**
::::

::::{dropdown} __Parameter:__ {ref}`Output verbosity<parameters:Adjoint/Optimization/Output_20verbosity>`
:name: parameters:Adjoint/Optimization/Output_20verbosity
**Default value:** 0

**Pattern:** [Integer range 0...2147483647 (inclusive)]

**Documentation:**
::::

::::{dropdown} __Parameter:__ {ref}`Step tolerance<parameters:Adjoint/Optimization/Step_20tolerance>`
:name: parameters:Adjoint/Optimization/Step_20tolerance
**Default value:** 1e-12

**Pattern:** [Double 0...MAX_DOUBLE (inclusive)]

**Documentation:**
::::

::::{dropdown} __Parameter:__ {ref}`Upper bound<parameters:Adjoint/Optimization/Upper_20bound>`
:name: parameters:Adjoint/Optimization/Upper_20bound
**Default value:** 1e-2

**Pattern:** [Double -MAX_DOUBLE...MAX_DOUBLE (inclusive)]

**Documentation:**
::::

::::{dropdown} __Parameter:__ {ref}`Use physical temperature bounds<parameters:Adjoint/Optimization/Use_20physical_20temperature_20bounds>`
:name: parameters:Adjoint/Optimization/Use_20physical_20temperature_20bounds
**Default value:** false

**Pattern:** [Bool]

**Documentation:** Use the per-control bounds induced by 0 <= T_fixed + Pm <= 1 instead of uniform increment bounds.
::::

(parameters:Adjoint/Self_2dgravitating_20geoid)=
## **Subsection:** Adjoint / Self-gravitating geoid
::::{dropdown} __Parameter:__ {ref}`CMB density below<parameters:Adjoint/Self_2dgravitating_20geoid/CMB_20density_20below>`
:name: parameters:Adjoint/Self_2dgravitating_20geoid/CMB_20density_20below
**Default value:** 9900

**Pattern:** [Double 0...MAX_DOUBLE (inclusive)]

**Documentation:** Fixed exterior density below the CMB. It must match Postprocess/Dynamic topography/Density below.
::::

::::{dropdown} __Parameter:__ {ref}`Coefficient weights<parameters:Adjoint/Self_2dgravitating_20geoid/Coefficient_20weights>`
:name: parameters:Adjoint/Self_2dgravitating_20geoid/Coefficient_20weights
**Default value:**

**Pattern:** [Anything]

**Documentation:** Optional nonnegative degree/order weights shared by all four coupled fields and by cosine/sine.
::::

::::{dropdown} __Parameter:__ {ref}`Maximum spherical harmonic degree<parameters:Adjoint/Self_2dgravitating_20geoid/Maximum_20spherical_20harmonic_20degree>`
:name: parameters:Adjoint/Self_2dgravitating_20geoid/Maximum_20spherical_20harmonic_20degree
**Default value:** 2

**Pattern:** [Integer range 2...2147483647 (inclusive)]

**Documentation:** Maximum coupled self-gravity harmonic degree.
::::

::::{dropdown} __Parameter:__ {ref}`Minimum spherical harmonic degree<parameters:Adjoint/Self_2dgravitating_20geoid/Minimum_20spherical_20harmonic_20degree>`
:name: parameters:Adjoint/Self_2dgravitating_20geoid/Minimum_20spherical_20harmonic_20degree
**Default value:** 2

**Pattern:** [Integer range 2...2147483647 (inclusive)]

**Documentation:** Minimum coupled self-gravity harmonic degree. The coupled operator is defined only for l >= 2.
::::

::::{dropdown} __Parameter:__ {ref}`Observed CMB geoid cosine coefficients<parameters:Adjoint/Self_2dgravitating_20geoid/Observed_20CMB_20geoid_20cosine_20coefficients>`
:name: parameters:Adjoint/Self_2dgravitating_20geoid/Observed_20CMB_20geoid_20cosine_20coefficients
**Default value:**

**Pattern:** [Anything]

**Documentation:** Observed coupled coefficient values in degree-major order.
::::

::::{dropdown} __Parameter:__ {ref}`Observed CMB geoid sine coefficients<parameters:Adjoint/Self_2dgravitating_20geoid/Observed_20CMB_20geoid_20sine_20coefficients>`
:name: parameters:Adjoint/Self_2dgravitating_20geoid/Observed_20CMB_20geoid_20sine_20coefficients
**Default value:**

**Pattern:** [Anything]

**Documentation:** Observed coupled coefficient values in degree-major order.
::::

::::{dropdown} __Parameter:__ {ref}`Observed CMB topography cosine coefficients<parameters:Adjoint/Self_2dgravitating_20geoid/Observed_20CMB_20topography_20cosine_20coefficients>`
:name: parameters:Adjoint/Self_2dgravitating_20geoid/Observed_20CMB_20topography_20cosine_20coefficients
**Default value:**

**Pattern:** [Anything]

**Documentation:** Observed coupled coefficient values in degree-major order.
::::

::::{dropdown} __Parameter:__ {ref}`Observed CMB topography sine coefficients<parameters:Adjoint/Self_2dgravitating_20geoid/Observed_20CMB_20topography_20sine_20coefficients>`
:name: parameters:Adjoint/Self_2dgravitating_20geoid/Observed_20CMB_20topography_20sine_20coefficients
**Default value:**

**Pattern:** [Anything]

**Documentation:** Observed coupled coefficient values in degree-major order.
::::

::::{dropdown} __Parameter:__ {ref}`Observed surface geoid cosine coefficients<parameters:Adjoint/Self_2dgravitating_20geoid/Observed_20surface_20geoid_20cosine_20coefficients>`
:name: parameters:Adjoint/Self_2dgravitating_20geoid/Observed_20surface_20geoid_20cosine_20coefficients
**Default value:**

**Pattern:** [Anything]

**Documentation:** Observed coupled coefficient values in degree-major order.
::::

::::{dropdown} __Parameter:__ {ref}`Observed surface geoid sine coefficients<parameters:Adjoint/Self_2dgravitating_20geoid/Observed_20surface_20geoid_20sine_20coefficients>`
:name: parameters:Adjoint/Self_2dgravitating_20geoid/Observed_20surface_20geoid_20sine_20coefficients
**Default value:**

**Pattern:** [Anything]

**Documentation:** Observed coupled coefficient values in degree-major order.
::::

::::{dropdown} __Parameter:__ {ref}`Observed surface topography cosine coefficients<parameters:Adjoint/Self_2dgravitating_20geoid/Observed_20surface_20topography_20cosine_20coefficients>`
:name: parameters:Adjoint/Self_2dgravitating_20geoid/Observed_20surface_20topography_20cosine_20coefficients
**Default value:**

**Pattern:** [Anything]

**Documentation:** Observed coupled coefficient values in degree-major order.
::::

::::{dropdown} __Parameter:__ {ref}`Observed surface topography sine coefficients<parameters:Adjoint/Self_2dgravitating_20geoid/Observed_20surface_20topography_20sine_20coefficients>`
:name: parameters:Adjoint/Self_2dgravitating_20geoid/Observed_20surface_20topography_20sine_20coefficients
**Default value:**

**Pattern:** [Anything]

**Documentation:** Observed coupled coefficient values in degree-major order.
::::

::::{dropdown} __Parameter:__ {ref}`Surface density above<parameters:Adjoint/Self_2dgravitating_20geoid/Surface_20density_20above>`
:name: parameters:Adjoint/Self_2dgravitating_20geoid/Surface_20density_20above
**Default value:** 0

**Pattern:** [Double 0...MAX_DOUBLE (inclusive)]

**Documentation:** Fixed exterior density above the surface. It must match Postprocess/Dynamic topography/Density above.
::::

(parameters:Adjoint/Temperature_20field)=
## **Subsection:** Adjoint / Temperature field
::::{dropdown} __Parameter:__ {ref}`Coordinate system<parameters:Adjoint/Temperature_20field/Coordinate_20system>`
:name: parameters:Adjoint/Temperature_20field/Coordinate_20system
**Default value:** cartesian

**Pattern:** [Selection cartesian|spherical|depth ]

**Documentation:** Coordinate system used to evaluate the terminal target and weight functions.
::::

::::{dropdown} __Parameter:__ {ref}`Observation source<parameters:Adjoint/Temperature_20field/Observation_20source>`
:name: parameters:Adjoint/Temperature_20field/Observation_20source
**Default value:** function

**Pattern:** [Selection function|file ]

**Documentation:** Reference-data source for this terminal scalar-field misfit. A file is read once and interpolated into the current ASPECT finite-element space.
::::

(parameters:Adjoint/Temperature_20field/Observation_20file)=
## **Subsection:** Adjoint / Temperature field / Observation file
::::{dropdown} __Parameter:__ {ref}`Data directory<parameters:Adjoint/Temperature_20field/Observation_20file/Data_20directory>`
:name: parameters:Adjoint/Temperature_20field/Observation_20file/Data_20directory
**Default value:** .

**Pattern:** [DirectoryName]

**Documentation:** The name of a directory that contains the model data. This path may either be absolute (if starting with a &lsquo;/&rsquo;) or relative to the current directory. The path may also include the special text &lsquo;$ASPECT_SOURCE_DIR&rsquo; which will be interpreted as the path in which the ASPECT source files were located when ASPECT was compiled. This interpretation allows, for example, to reference files located in the &lsquo;data/&rsquo; subdirectory of ASPECT. A trailing slash at the end of the directory path is optional; the plugin will automatically append a &rsquo;/&rsquo; when the parameters are parsed if it is missing.
::::

::::{dropdown} __Parameter:__ {ref}`Data file name<parameters:Adjoint/Temperature_20field/Observation_20file/Data_20file_20name>`
:name: parameters:Adjoint/Temperature_20field/Observation_20file/Data_20file_20name
**Default value:** observation.txt

**Pattern:** [Anything]

**Documentation:** The file name of the model data.
::::

::::{dropdown} __Parameter:__ {ref}`First point on slice<parameters:Adjoint/Temperature_20field/Observation_20file/First_20point_20on_20slice>`
:name: parameters:Adjoint/Temperature_20field/Observation_20file/First_20point_20on_20slice
**Default value:** 0.0,1.0,0.0

**Pattern:** [Anything]

**Documentation:** Point that determines the plane in which the 2d slice lies in. This variable is only used if &rsquo;Slice dataset in 2d plane&rsquo; is true. The slice will go through this point, the point defined by the parameter &rsquo;Second point on slice&rsquo;, and the center of the model domain. After the rotation, this first point will lie along the (0,1,0) axis of the coordinate system. The coordinates of the point have to be given in Cartesian coordinates.
::::

::::{dropdown} __Parameter:__ {ref}`Scale factor<parameters:Adjoint/Temperature_20field/Observation_20file/Scale_20factor>`
:name: parameters:Adjoint/Temperature_20field/Observation_20file/Scale_20factor
**Default value:** 1.

**Pattern:** [Double -MAX_DOUBLE...MAX_DOUBLE (inclusive)]

**Documentation:** Scalar factor, which is applied to the model data. You might want to use this to scale the input to a reference model. Another way to use this factor is to convert units of the input files. For instance, if you provide velocities in cm/yr set this factor to 0.01.
::::

::::{dropdown} __Parameter:__ {ref}`Second point on slice<parameters:Adjoint/Temperature_20field/Observation_20file/Second_20point_20on_20slice>`
:name: parameters:Adjoint/Temperature_20field/Observation_20file/Second_20point_20on_20slice
**Default value:** 1.0,0.0,0.0

**Pattern:** [Anything]

**Documentation:** Second point that determines the plane in which the 2d slice lies in. This variable is only used if &rsquo;Slice dataset in 2d plane&rsquo; is true. The slice will go through this point, the point defined by the parameter &rsquo;First point on slice&rsquo;, and the center of the model domain. The coordinates of the point have to be given in Cartesian coordinates.
::::

::::{dropdown} __Parameter:__ {ref}`Slice dataset in 2D plane<parameters:Adjoint/Temperature_20field/Observation_20file/Slice_20dataset_20in_202D_20plane>`
:name: parameters:Adjoint/Temperature_20field/Observation_20file/Slice_20dataset_20in_202D_20plane
**Default value:** false

**Pattern:** [Bool]

**Documentation:** Whether to use a 2d data slice of a 3d data file or the entire data file. Slicing a 3d dataset is only supported for 2d models.
::::

(parameters:Adjoint/Temperature_20field/Target_20function)=
## **Subsection:** Adjoint / Temperature field / Target function
::::{dropdown} __Parameter:__ {ref}`Function constants<parameters:Adjoint/Temperature_20field/Target_20function/Function_20constants>`
:name: parameters:Adjoint/Temperature_20field/Target_20function/Function_20constants
**Default value:**

**Pattern:** [Anything]

**Documentation:** Sometimes it is convenient to use symbolic constants in the expression that describes the function, rather than having to use its numeric value everywhere the constant appears. These values can be defined using this parameter, in the form &lsquo;var1=value1, var2=value2, ...&rsquo;.

A typical example would be to set this runtime parameter to &lsquo;pi=3.1415926536&rsquo; and then use &lsquo;pi&rsquo; in the expression of the actual formula. (That said, for convenience this class actually defines both &lsquo;pi&rsquo; and &lsquo;Pi&rsquo; by default, but you get the idea.)
::::

::::{dropdown} __Parameter:__ {ref}`Function expression<parameters:Adjoint/Temperature_20field/Target_20function/Function_20expression>`
:name: parameters:Adjoint/Temperature_20field/Target_20function/Function_20expression
**Default value:** 0

**Pattern:** [Anything]

**Documentation:** Target final scalar field.
::::

::::{dropdown} __Parameter:__ {ref}`Variable names<parameters:Adjoint/Temperature_20field/Target_20function/Variable_20names>`
:name: parameters:Adjoint/Temperature_20field/Target_20function/Variable_20names
**Default value:** x,y,t

**Pattern:** [Anything]

**Documentation:** The names of the variables as they will be used in the function, separated by commas. By default, the names of variables at which the function will be evaluated are &lsquo;x&rsquo; (in 1d), &lsquo;x,y&rsquo; (in 2d) or &lsquo;x,y,z&rsquo; (in 3d) for spatial coordinates and &lsquo;t&rsquo; for time. You can then use these variable names in your function expression and they will be replaced by the values of these variables at which the function is currently evaluated. However, you can also choose a different set of names for the independent variables at which to evaluate your function expression. For example, if you work in spherical coordinates, you may wish to set this input parameter to &lsquo;r,phi,theta,t&rsquo; and then use these variable names in your function expression.
::::

(parameters:Adjoint/Temperature_20field/Weight_20function)=
## **Subsection:** Adjoint / Temperature field / Weight function
::::{dropdown} __Parameter:__ {ref}`Function constants<parameters:Adjoint/Temperature_20field/Weight_20function/Function_20constants>`
:name: parameters:Adjoint/Temperature_20field/Weight_20function/Function_20constants
**Default value:**

**Pattern:** [Anything]

**Documentation:** Sometimes it is convenient to use symbolic constants in the expression that describes the function, rather than having to use its numeric value everywhere the constant appears. These values can be defined using this parameter, in the form &lsquo;var1=value1, var2=value2, ...&rsquo;.

A typical example would be to set this runtime parameter to &lsquo;pi=3.1415926536&rsquo; and then use &lsquo;pi&rsquo; in the expression of the actual formula. (That said, for convenience this class actually defines both &lsquo;pi&rsquo; and &lsquo;Pi&rsquo; by default, but you get the idea.)
::::

::::{dropdown} __Parameter:__ {ref}`Function expression<parameters:Adjoint/Temperature_20field/Weight_20function/Function_20expression>`
:name: parameters:Adjoint/Temperature_20field/Weight_20function/Function_20expression
**Default value:** 1

**Pattern:** [Anything]

**Documentation:** Non-negative spatial weight for the terminal scalar-field misfit.
::::

::::{dropdown} __Parameter:__ {ref}`Variable names<parameters:Adjoint/Temperature_20field/Weight_20function/Variable_20names>`
:name: parameters:Adjoint/Temperature_20field/Weight_20function/Variable_20names
**Default value:** x,y,t

**Pattern:** [Anything]

**Documentation:** The names of the variables as they will be used in the function, separated by commas. By default, the names of variables at which the function will be evaluated are &lsquo;x&rsquo; (in 1d), &lsquo;x,y&rsquo; (in 2d) or &lsquo;x,y,z&rsquo; (in 3d) for spatial coordinates and &lsquo;t&rsquo; for time. You can then use these variable names in your function expression and they will be replaced by the values of these variables at which the function is currently evaluated. However, you can also choose a different set of names for the independent variables at which to evaluate your function expression. For example, if you work in spherical coordinates, you may wish to set this input parameter to &lsquo;r,phi,theta,t&rsquo; and then use these variable names in your function expression.
::::

(parameters:Adjoint/Time_20dependent)=
## **Subsection:** Adjoint / Time dependent
::::{dropdown} __Parameter:__ {ref}`Adjoint checkpoint interval<parameters:Adjoint/Time_20dependent/Adjoint_20checkpoint_20interval>`
:name: parameters:Adjoint/Time_20dependent/Adjoint_20checkpoint_20interval
**Default value:** 50

**Pattern:** [Integer range 1...2147483647 (inclusive)]

**Documentation:** Number of completed reverse steps between durable adjoint restart checkpoints. This is independent of ASPECT&rsquo;s forward checkpoint interval.
::::

::::{dropdown} __Parameter:__ {ref}`Advection preconditioner<parameters:Adjoint/Time_20dependent/Advection_20preconditioner>`
:name: parameters:Adjoint/Time_20dependent/Advection_20preconditioner
**Default value:** transpose ilu

**Pattern:** [Selection none|forward ilu experimental|transpose ilu ]

**Documentation:** Preconditioner for the time-dependent transposed advection solve. This affects only Krylov convergence, not the discrete adjoint equation. The default rebuilds ILU from an explicit A^T matrix for every restored snapshot.
::::

::::{dropdown} __Parameter:__ {ref}`Trajectory anchor spacing<parameters:Adjoint/Time_20dependent/Trajectory_20anchor_20spacing>`
:name: parameters:Adjoint/Time_20dependent/Trajectory_20anchor_20spacing
**Default value:** 1

**Pattern:** [Integer range 1...2147483647 (inclusive)]

**Documentation:** Fixed Phase-B sparse replay anchor spacing. A value of 1 retains the dense full-archive behavior; larger values retain only three-level anchor packages while preserving the dense accepted-step schedule journal.
::::

::::{dropdown} __Parameter:__ {ref}`Trajectory cleanup policy<parameters:Adjoint/Time_20dependent/Trajectory_20cleanup_20policy>`
:name: parameters:Adjoint/Time_20dependent/Trajectory_20cleanup_20policy
**Default value:** safe incremental

**Pattern:** [Selection safe incremental|on success|keep ]

**Documentation:** Lifetime policy for the temporary time-dependent adjoint trajectory. &rsquo;safe incremental&rsquo; writes and commits an independent reverse checkpoint before reclaiming obsolete records. &rsquo;on success&rsquo; retains all temporary data until a successful reverse sweep. &rsquo;keep&rsquo; never removes it automatically.
::::

::::{dropdown} __Parameter:__ {ref}`Trajectory records per segment<parameters:Adjoint/Time_20dependent/Trajectory_20records_20per_20segment>`
:name: parameters:Adjoint/Time_20dependent/Trajectory_20records_20per_20segment
**Default value:** 4

**Pattern:** [Integer range 1...2147483647 (inclusive)]

**Documentation:** Number of logical trajectory records in each shared MPI-I/O data file. This archive-layout setting must match when reopening an adjoint trajectory.
::::

::::{dropdown} __Parameter:__ {ref}`Trajectory storage mode<parameters:Adjoint/Time_20dependent/Trajectory_20storage_20mode>`
:name: parameters:Adjoint/Time_20dependent/Trajectory_20storage_20mode
**Default value:** minimal

**Pattern:** [Selection minimal|stokes cached ]

**Documentation:** Storage policy for the time-dependent adjoint trajectory. &rsquo;minimal&rsquo; archives only temperature and composition and recovers Stokes fields during the reverse sweep. &rsquo;stokes cached&rsquo; additionally archives velocity and pressure. This is not the ASPECT restart-checkpoint mechanism.
::::
