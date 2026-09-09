! Thin shim exposing ACDC's steady-state solve with an f2py-friendly
! signature.
!
! This is NEW code belonging to acdc-jax, not a modification of the vendored
! reference -- fortran/ stays byte-identical to upstream. It exists because
! acdc_plugin takes a `character(len=11), dimension(n_neutral_monomers)`
! array of vapour names, which f2py wraps badly, and two optional arguments.
!
! The body mirrors fortran/src/run_acdc_J_example.f90 exactly, including the
! 60 s / 0 s time arguments, which are dummies under the steady-state
! assumption: get_acdc_J.f90:99-104 overrides the simulation time with
! 1e8 s whenever solve_ss is set, as it is here.
!
! IMPORTANT: acdc_plugin keeps the concentration vector in a `save`d array
! (get_acdc_J.f90:31) and warm-starts from it, so consecutive calls in one
! process are NOT independent. Callers that need independent results must
! use a fresh process per condition. See validation/capture_steadystate.py,
! which measures how much this actually matters.

subroutine solve_j(c_vapor_a, c_vapor_n, cs_ref, temp, ipr, j_acdc, diameter_acdc)

    use get_acdc_J, only : acdc_plugin

    implicit none

    ! Input, all SI
    real(kind(1.d0)), intent(in) :: c_vapor_a      ! [H2SO4], m^-3
    real(kind(1.d0)), intent(in) :: c_vapor_n      ! [NH3], m^-3
    real(kind(1.d0)), intent(in) :: cs_ref         ! reference coagulation sink, 1/s
    real(kind(1.d0)), intent(in) :: temp           ! temperature, K
    real(kind(1.d0)), intent(in) :: ipr            ! ion production rate, 1/m^3/s

    ! Output
    real(kind(1.d0)), intent(out) :: j_acdc        ! formation rate, 1/m^3/s
    real(kind(1.d0)), intent(out) :: diameter_acdc ! mass diameter of formed particles, m

    character(len=11), dimension(2) :: names_vapor
    real(kind(1.d0)) :: c_vapor(2)

    names_vapor(1)(:) = 'A'
    names_vapor(2)(:) = 'N'

    ! acdc_plugin takes c_vapor as intent(inout) and writes the vapour
    ! concentrations back, so it must be a local copy rather than the
    ! caller's input.
    c_vapor = (/c_vapor_a, c_vapor_n/)

    call acdc_plugin(names_vapor, c_vapor, cs_ref, temp, ipr, &
        & 60.d0, 0.d0, j_acdc, diameter_acdc)

end subroutine solve_j
